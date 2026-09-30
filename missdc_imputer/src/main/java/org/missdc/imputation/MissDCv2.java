package org.missdc.imputation;

import it.unimi.dsi.fastutil.floats.Float2ObjectMap;
import it.unimi.dsi.fastutil.floats.FloatList;
import org.missdc.dc.predicates.IndexedPredicate;
import org.missdc.dc.predicates.groups.ColumnPredicateGroup;
import org.missdc.dc.predicates.groups.NumericalColumnPredicateGroup;
import org.missdc.dc.predicates.indexes.EqualityIndex;
import org.missdc.dc.predicates.indexes.LtBinnedIndex;
import org.missdc.dc.predicates.indexes.LtIndex;
import org.missdc.dc.predicates.space.PredicateSpace;
import org.missdc.dc.predicates.space.builder.PredicateSpaceBuilder;
import org.missdc.discovery.DCDiscoverer;
import org.missdc.discovery.enumeration.HybridEvidenceInversion;
import org.missdc.imputation.voting.VotingStrategy;
import org.missdc.input.Table;
import org.missdc.input.columns.CategoricalColumn;
import org.missdc.input.columns.Column;
import org.missdc.input.reader.CSVInput;
import org.missdc.utils.bitset.BitUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;

/**
 * MissDC v2: the stable evolution of {@link MissDC}.
 *
 * <p>Defaults:</p>
 * <ul>
 *   <li>minimum evidence multiplicity = 8;</li>
 *   <li>maximum DC size = 7;</li>
 *   <li>voting = CURRENT, i.e. every (DC, donor tuple) occurrence votes;</li>
 *   <li>exact DC discovery, no ML fallback.</li>
 * </ul>
 *
 */
public final class MissDCv2 {

    public static final long DEFAULT_MIN_EVIDENCE_MULTIPLICITY = 8L;
    public static final int DEFAULT_MAX_DC_SIZE = 7;
    public static final VotingStrategy DEFAULT_VOTING_STRATEGY = VotingStrategy.TUPLE;

    private static final Logger log = LoggerFactory.getLogger(MissDCv2.class);
    private static final int BIN_THRESHOLD = 2000;
    private static final int UNBOUNDED_DC_SIZE = HybridEvidenceInversion.UNBOUNDED;

    private final String dirtyDatasetPath;
    private final boolean useApproximateDCs;
    private final long minEvidenceMultiplicity;
    private final int maxDCSize;
    private final VotingStrategy votingStrategy;

    private PredicateSpace predicateSpace;
    private int[] directionalToUneq;
    private Map<Integer, Set<Integer>> tid2NullColMap;
    private Map<BitSet, ArrayList<Integer>> dcs2ExecutionOrder;
    private String[][] imputedDataset;

    private int discoveredDCCount;
    private int filteredDCCount;
    private long discoveryTimeMs;
    private long imputationTimeMs;
    private final LongAdder refinementVotingNanos = new LongAdder();
    private final LongAdder missingCells = new LongAdder();
    private final LongAdder repairedCells = new LongAdder();

    public MissDCv2(String dirtyDatasetPath) {
        this(
                dirtyDatasetPath,
                false,
                DEFAULT_MIN_EVIDENCE_MULTIPLICITY,
                DEFAULT_MAX_DC_SIZE,
                DEFAULT_VOTING_STRATEGY);
    }

    public MissDCv2(
            String dirtyDatasetPath,
            boolean useApproximateDCs,
            long minEvidenceMultiplicity,
            int maxDCSize,
            VotingStrategy votingStrategy) {

        if (dirtyDatasetPath == null || dirtyDatasetPath.isBlank()) {
            throw new IllegalArgumentException("dirtyDatasetPath cannot be blank");
        }
        if (votingStrategy == null) {
            throw new IllegalArgumentException("votingStrategy cannot be null");
        }
        if (minEvidenceMultiplicity < 1) {
            throw new IllegalArgumentException("minEvidenceMultiplicity must be >= 1");
        }
        if (maxDCSize != UNBOUNDED_DC_SIZE && maxDCSize < 1) {
            throw new IllegalArgumentException("maxDCSize must be >= 1 or UNBOUNDED (-1)");
        }
        if (useApproximateDCs && maxDCSize != UNBOUNDED_DC_SIZE) {
            throw new IllegalArgumentException(
                    "Bounded DC enumeration is supported only for exact DC discovery. "
                            + "Use maxDCSize=-1 for approximate DCs.");
        }

        this.dirtyDatasetPath = dirtyDatasetPath;
        this.useApproximateDCs = useApproximateDCs;
        this.minEvidenceMultiplicity = minEvidenceMultiplicity;
        this.maxDCSize = maxDCSize;
        this.votingStrategy = votingStrategy;
    }

    public void run() throws Exception {
        resetStatistics();

        log.info(
                "Running MissDCv2: voting={}, minEvidenceMultiplicity={}, maxDCSize={} ...",
                votingStrategy,
                minEvidenceMultiplicity,
                maxDCSize == UNBOUNDED_DC_SIZE ? "max" : maxDCSize);

        long discoveryStart = System.nanoTime();
        Set<BitSet> dcs = discoverDCs();
        discoveryTimeMs = nanosToMillis(System.nanoTime() - discoveryStart);
        discoveredDCCount = dcs.size();

        runImputation(dcs);
    }

    /**
     * Runs the production MissDCv2 imputation pipeline with an externally
     * supplied DC set. This is intended for controlled experiments where the
     * DC source is the independent variable (for example exact vs approximate
     * vs LIMA discovery). The supplied predicates must use the same predicate
     * identifiers produced by {@link PredicateSpaceBuilder} for this dataset.
     *
     * <p>Discovery is deliberately not performed here, so
     * {@link RunStatistics#discoveryTimeMs()} is zero. The caller should time
     * the external DC source separately.</p>
     */
    public void runWithDCs(Set<BitSet> inputDCs) throws Exception {
        if (inputDCs == null) {
            throw new IllegalArgumentException("inputDCs cannot be null");
        }

        resetStatistics();
        Set<BitSet> dcs = cloneDCSet(inputDCs);
        discoveredDCCount = dcs.size();
        discoveryTimeMs = 0L;

        log.info(
                "Running MissDCv2 with external DC input: voting={}, inputDCs={} ...",
                votingStrategy,
                discoveredDCCount);

        runImputation(dcs);
    }

    private void runImputation(Set<BitSet> dcs) throws Exception {
        long imputationStart = System.nanoTime();

        CSVInput dirtyInput = new CSVInput(dirtyDatasetPath);
        Table dirtyTable = dirtyInput.getTable();
        imputedDataset = initializeDataset(dirtyTable);
        predicateSpace = new PredicateSpaceBuilder().build(dirtyTable);

        buildDirectionalToUneqMap();
        int beforeFilter = dcs.size();
        dcs = filterRedundantDirectionalDCs(dcs);
        filteredDCCount = dcs.size();
        log.info(
                "Directional redundancy filter: {} -> {} DCs ({} removed)",
                beforeFilter,
                filteredDCCount,
                beforeFilter - filteredDCCount);

        List<Column> colsWithNulls = dirtyTable.getColsWithNulls();
        buildTID2NullColMap(colsWithNulls);
        buildPredicateOrder4Dcs(dcs);
        buildPredicateIndexes(predicateSpace, dirtyTable);

        // Keep MissDC's low-cardinality-first sequential column order.
        colsWithNulls.sort(Comparator.comparingLong(Column::getCardinality));
        for (Column col : colsWithNulls) {
            int numMissingInitial = col.getTIDsMissing().size();
            if (numMissingInitial == 0) {
                continue;
            }
            missingCells.add(numMissingInitial);

            MissDCv2VotingTree refinementTree = buildRefinementTree(dirtyTable, dcs, col);
            Map<Integer, Float> repairs4col = new ConcurrentHashMap<>();

            if (col.getDomain().size() == 1) {
                Float repair = col.getDomain().iterator().next();
                for (Integer tid : col.getTIDsMissing()) {
                    repairs4col.put(tid, repair);
                    repairedCells.increment();
                }

            } else if (refinementTree.getTrackedDCCount() > 0) {
                long refinementStart = System.nanoTime();
                int cores = Math.max(1, Runtime.getRuntime().availableProcessors());
                ExecutorService service = Executors.newFixedThreadPool(cores);

                for (Integer tid : col.getTIDsMissing()) {
                    service.submit(() -> {
                        Set<Integer> nullifiedCols = tid2NullColMap.get(tid);
                        if (nullifiedCols == null) {
                            nullifiedCols = Set.of(col.ColumnIndex);
                        }

                        MissDCv2VotingTree.Selection selection =
                                refinementTree.refineAndVote(tid, nullifiedCols, votingStrategy);

                        if (selection.hasCandidate()) {
                            repairs4col.put(tid, selection.candidate());
                            repairedCells.increment();
                        }
                    });
                }

                service.shutdown();
                try {
                    service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(
                            "Interrupted while waiting for MissDCv2 workers", e);
                }
                refinementVotingNanos.add(System.nanoTime() - refinementStart);
            }

            applyImputations(dirtyTable, col, repairs4col);
        }

        dirtyInput.saveImputed(imputedDataset);
        imputationTimeMs = nanosToMillis(System.nanoTime() - imputationStart);

        RunStatistics stats = getStatistics();
        log.info(
                "MissDCv2 finished: voting={}, discoveredDCs={}, filteredDCs={}, repaired={}/{}, "
                        + "discoveryMs={}, imputationMs={}, refinementVotingMs={}",
                stats.votingStrategy(),
                stats.discoveredDCs(),
                stats.filteredDCs(),
                stats.repairedCells(),
                stats.missingCells(),
                stats.discoveryTimeMs(),
                stats.imputationTimeMs(),
                stats.refinementVotingTimeMs());
    }

    public RunStatistics getStatistics() {
        return new RunStatistics(
                votingStrategy,
                minEvidenceMultiplicity,
                maxDCSize,
                discoveredDCCount,
                filteredDCCount,
                missingCells.sum(),
                repairedCells.sum(),
                discoveryTimeMs,
                imputationTimeMs,
                nanosToMillis(refinementVotingNanos.sum()));
    }

    private Set<BitSet> discoverDCs() throws Exception {
        DCDiscoverer discoverer = new DCDiscoverer(dirtyDatasetPath, useApproximateDCs);
        return discoverer.run(minEvidenceMultiplicity, maxDCSize);
    }

    private MissDCv2VotingTree buildRefinementTree(
            Table dirtyTable,
            Set<BitSet> dcs,
            Column col) {

        int targetUneqId = predicateSpace
                .getColumn2GroupMap()
                .get(col)
                .getUneq()
                .getPredicateId();

        Set<BitSet> dcsForColumn = BitUtils.getSubSetWithId(dcs, targetUneqId);
        MissDCv2VotingTree tree = new MissDCv2VotingTree(dirtyTable, predicateSpace, col);

        for (BitSet dc : dcsForColumn) {
            ArrayList<Integer> path = new ArrayList<>(dcs2ExecutionOrder.get(dc));
            path.remove((Integer) targetUneqId);
            tree.addDC(path);
        }

        log.info(
                "MissDCv2 tree for column {}: {} target DCs",
                col.ColumnName,
                dcsForColumn.size());
        return tree;
    }

    private void buildDirectionalToUneqMap() {
        directionalToUneq = new int[predicateSpace.size()];
        Arrays.fill(directionalToUneq, -1);

        for (int pid = 0; pid < predicateSpace.size(); pid++) {
            IndexedPredicate predicate = predicateSpace.getPredicateById(pid);
            if (!predicate.isLT() && !predicate.isGT()) {
                continue;
            }

            directionalToUneq[pid] = predicateSpace
                    .getColumn2GroupMap()
                    .get(predicate.getPredicate().getCol1())
                    .getUneq()
                    .getPredicateId();
        }
    }

    private Set<BitSet> filterRedundantDirectionalDCs(Set<BitSet> dcs) {
        if (dcs.size() < 2) {
            return dcs;
        }

        Set<BitSet> dcIndex = dcs instanceof HashSet ? dcs : new HashSet<>(dcs);
        Set<BitSet> filtered = new LinkedHashSet<>(dcs.size());
        BitSet probe = new BitSet(predicateSpace.size() / 2);

        for (BitSet dc : dcs) {
            boolean redundant = false;
            probe.clear();
            probe.or(dc);

            for (int pid = dc.nextSetBit(0); pid >= 0; pid = dc.nextSetBit(pid + 1)) {
                int uneqPid = directionalToUneq[pid];
                if (uneqPid < 0 || dc.get(uneqPid)) {
                    continue;
                }

                probe.clear(pid);
                probe.set(uneqPid);

                if (dcIndex.contains(probe)) {
                    redundant = true;
                    break;
                }

                probe.clear(uneqPid);
                probe.set(pid);
            }

            if (!redundant) {
                filtered.add(dc);
            }
        }

        return filtered;
    }

    private void buildTID2NullColMap(List<Column> colsWithNulls) {
        tid2NullColMap = new LinkedHashMap<>();
        for (Column col : colsWithNulls) {
            for (Integer tid : col.getTIDsMissing()) {
                tid2NullColMap
                        .computeIfAbsent(tid, ignored -> new HashSet<>())
                        .add(col.ColumnIndex);
            }
        }
    }

    private void buildPredicateOrder4Dcs(Set<BitSet> dcs) {
        dcs2ExecutionOrder = new HashMap<>();

        for (BitSet dc : dcs) {
            List<Integer> eqs = new ArrayList<>();
            List<Integer> uneqs = new ArrayList<>();
            List<Integer> ranges = new ArrayList<>();

            for (int pid = dc.nextSetBit(0); pid >= 0; pid = dc.nextSetBit(pid + 1)) {
                IndexedPredicate pred = predicateSpace.getPredicateById(pid);
                if (pred.isEQ()) {
                    eqs.add(pid);
                } else if (pred.isUNEQ()) {
                    uneqs.add(pid);
                } else {
                    ranges.add(pid);
                }
            }

            Comparator<Integer> byCardinality = Comparator.comparingLong(
                    pid -> predicateSpace
                            .getPredicateById(pid)
                            .getPredicate()
                            .getCol1()
                            .getCardinality());

            eqs.sort(byCardinality);
            ranges.sort(byCardinality);
            uneqs.sort(byCardinality);
            Collections.reverse(eqs);
            Collections.reverse(ranges);
            Collections.reverse(uneqs);

            ArrayList<Integer> executionOrder = new ArrayList<>();
            executionOrder.addAll(eqs);
            executionOrder.addAll(ranges);
            executionOrder.addAll(uneqs);
            dcs2ExecutionOrder.put(dc, executionOrder);
        }
    }

    private void buildPredicateIndexes(PredicateSpace predicateSpace, Table table) {
        for (ColumnPredicateGroup catGroup : predicateSpace.getCategoricalPredicateGroups()) {
            IndexedPredicate eq = catGroup.getEq();
            EqualityIndex eqIndex = new EqualityIndex(eq);
            eq.setEqualityIndex(eqIndex);
        }

        for (NumericalColumnPredicateGroup numGroup : predicateSpace.getNumericalPredicateGroups()) {
            IndexedPredicate eq = numGroup.getEq();
            EqualityIndex eqIndex = new EqualityIndex(eq);
            eq.setEqualityIndex(eqIndex);

            if (eqIndex.getIndex().size() >= BIN_THRESHOLD) {
                numGroup.setBinnedLT(true);
                LtBinnedIndex ltBinnedIndex = new LtBinnedIndex(eqIndex, table.getNUM_RECORDS());
                numGroup.getLt().setLtBinnedIndex(ltBinnedIndex);
            } else {
                LtIndex ltIndex = new LtIndex(eqIndex);
                numGroup.getLt().setLtIndex(ltIndex);
            }
        }
    }

    private void applyImputations(Table dirtyTable, Column col, Map<Integer, Float> repairs4col) {
        FloatList values = dirtyTable.getColumnByName(col.ColumnName).getValuesList();

        for (Map.Entry<Integer, Float> entry : repairs4col.entrySet()) {
            int tid = entry.getKey();
            float repair = entry.getValue();
            values.set(tid, repair);

            if (col instanceof CategoricalColumn catCol) {
                Float2ObjectMap<String> reverseMapDirty = catCol.buildReverseDictionaryMap();
                imputedDataset[tid][col.ColumnIndex] = reverseMapDirty.get(repair);
            } else {
                imputedDataset[tid][col.ColumnIndex] = "" + repair;
            }
        }

        col.rebuildTIDsMissing();

        ColumnPredicateGroup group = predicateSpace.getColumn2GroupMap().get(col);
        if (group instanceof NumericalColumnPredicateGroup numGroup) {
            IndexedPredicate eq = numGroup.getEq();
            EqualityIndex eqIndex = new EqualityIndex(eq);
            eq.setEqualityIndex(eqIndex);

            if (eqIndex.getIndex().size() >= BIN_THRESHOLD) {
                numGroup.setBinnedLT(true);
                LtBinnedIndex ltBinnedIndex = new LtBinnedIndex(eqIndex, dirtyTable.getNUM_RECORDS());
                numGroup.getLt().setLtBinnedIndex(ltBinnedIndex);
            } else {
                LtIndex ltIndex = new LtIndex(eqIndex);
                numGroup.getLt().setLtIndex(ltIndex);
            }
        } else {
            IndexedPredicate eq = group.getEq();
            EqualityIndex eqIndex = new EqualityIndex(eq);
            eq.setEqualityIndex(eqIndex);
        }
    }

    private String[][] initializeDataset(Table dirtyTable) {
        String[][] output =
                new String[dirtyTable.getNUM_RECORDS()][dirtyTable.NUM_ORIGINAL_COLLUMNS];

        for (int rowId = 0; rowId < dirtyTable.getNUM_RECORDS(); rowId++) {
            for (int colId = 0; colId < dirtyTable.NUM_ORIGINAL_COLLUMNS; colId++) {
                Column col = dirtyTable.getAllColumns().get(colId);

                if (col instanceof CategoricalColumn catCol) {
                    Float2ObjectMap<String> reverseMap = catCol.buildReverseDictionaryMap();
                    float value = col.getValueAt(rowId);
                    String stringValue = reverseMap.get(value);
                    output[rowId][colId] = Column.DEFAULT_NULL_STRING.equals(stringValue)
                            ? null
                            : stringValue;
                } else {
                    float value = col.getValueAt(rowId);
                    output[rowId][colId] = Float.compare(value, Column.DEFAULT_NULL_NUMBER) == 0
                            ? null
                            : "" + value;
                }
            }
        }

        return output;
    }

    private static Set<BitSet> cloneDCSet(Set<BitSet> source) {
        Set<BitSet> copy = new LinkedHashSet<>(source.size());
        for (BitSet dc : source) {
            copy.add((BitSet) dc.clone());
        }
        return copy;
    }

    private void resetStatistics() {
        discoveredDCCount = 0;
        filteredDCCount = 0;
        discoveryTimeMs = 0L;
        imputationTimeMs = 0L;
        refinementVotingNanos.reset();
        missingCells.reset();
        repairedCells.reset();
    }

    private static long nanosToMillis(long nanos) {
        return nanos / 1_000_000L;
    }

    public record RunStatistics(
            VotingStrategy votingStrategy,
            long minEvidenceMultiplicity,
            int maxDCSize,
            int discoveredDCs,
            int filteredDCs,
            long missingCells,
            long repairedCells,
            long discoveryTimeMs,
            long imputationTimeMs,
            long refinementVotingTimeMs) {}
}