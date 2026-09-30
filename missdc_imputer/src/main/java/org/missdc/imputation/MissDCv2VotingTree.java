package org.missdc.imputation;

import org.missdc.dc.predicates.IndexedPredicate;
import org.missdc.dc.predicates.groups.NumericalColumnPredicateGroup;
import org.missdc.dc.predicates.space.PredicateSpace;
import org.missdc.imputation.voting.VotingStrategy;
import org.missdc.input.Table;
import org.missdc.input.columns.Column;
import org.roaringbitmap.RoaringBitmap;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Immutable refinement trie used by {@link MissDCv2}.
 *
 * <p>The tree keeps the production MissDC predicate ordering, but removes the
 * per-cell tree copy and avoids cloning a parent bitmap for each branch. At a
 * branch, each child is produced directly from the parent through RoaringBitmap
 * intersection/difference operations. Along a single-child chain, the bitmap is
 * safely refined in place.</p>
 *
 * <p>No predicate cache, root-voted reduction, or experimental trie reordering
 * is used here.</p>
 */
final class MissDCv2VotingTree {

    record Selection(Float candidate, long winningSupport, long totalVotes) {
        boolean hasCandidate() {
            return candidate != null;
        }
    }

    private static final class TrieNode {
        final int predicateId;
        final Refiner refiner;
        final List<TrieNode> children = new ArrayList<>();
        final Map<Integer, TrieNode> childrenByPredicate = new HashMap<>();
        int terminalDCCount;

        TrieNode(int predicateId, Refiner refiner) {
            this.predicateId = predicateId;
            this.refiner = refiner;
        }
    }

    private final PredicateSpace predicateSpace;
    private final Column targetColumn;
    private final RoaringBitmap initialTids;
    private final TrieNode root = new TrieNode(-1, null);

    private int trackedDCCount;

    MissDCv2VotingTree(Table table, PredicateSpace predicateSpace, Column targetColumn) {
        this.predicateSpace = predicateSpace;
        this.targetColumn = targetColumn;

        initialTids = new RoaringBitmap();
        initialTids.add(0L, (long) table.getNUM_RECORDS());
        for (Integer tid : targetColumn.getTIDsMissing()) {
            initialTids.remove(tid);
        }
    }

    void addDC(List<Integer> predicateIds) {
        TrieNode current = root;

        for (int predicateId : predicateIds) {
            TrieNode next = current.childrenByPredicate.get(predicateId);
            if (next == null) {
                IndexedPredicate predicate = predicateSpace.getPredicateById(predicateId);
                next = new TrieNode(predicateId, new Refiner(predicate));
                current.childrenByPredicate.put(predicateId, next);
                current.children.add(next);
            }
            current = next;
        }

        current.terminalDCCount++;
        trackedDCCount++;
    }

    int getTrackedDCCount() {
        return trackedDCCount;
    }

    Selection refineAndVote(
            int tid,
            Set<Integer> nullifiedColsForTid,
            VotingStrategy strategy) {

        Set<Integer> nullCols = nullifiedColsForTid == null
                ? Set.of()
                : nullifiedColsForTid;

        RoaringBitmap universe = initialTids.clone();
        universe.remove(tid);

        VoteState votes = new VoteState(targetColumn, strategy);

        // Production MissDC never observes an empty refinement path (a DC that
        // consists only of target UNEQ). Preserve that behavior for CURRENT so
        // default MissDCv2 remains compatible with MissDC.
        if (strategy != VotingStrategy.CURRENT && root.terminalDCCount > 0) {
            votes.observe(universe, root.terminalDCCount);
        }

        visitChildren(root, universe, tid, nullCols, strategy, votes);
        return votes.finish();
    }

    private void visitChildren(
            TrieNode parent,
            RoaringBitmap parentTids,
            int currentTuple,
            Set<Integer> currentNullCols,
            VotingStrategy strategy,
            VoteState votes) {

        List<TrieNode> children = parent.children;
        int childCount = children.size();

        if (childCount == 0 || parentTids.isEmpty()) {
            return;
        }

        if (childCount == 1) {
            TrieNode child = children.get(0);
            if (predicateUnavailable(child, currentNullCols)) {
                return;
            }

            // No sibling needs the parent bitmap, so mutate it in place.
            child.refiner.refine(predicateSpace, parentTids, currentTuple);
            if (!parentTids.isEmpty()) {
                visitAfterPredicate(
                        child,
                        parentTids,
                        currentTuple,
                        currentNullCols,
                        strategy,
                        votes);
            }
            return;
        }

        // Branching point: produce every child bitmap directly from the same
        // parent. This is the retained direct-bitmap-intersection optimization.
        for (TrieNode child : children) {
            if (predicateUnavailable(child, currentNullCols)) {
                continue;
            }

            RoaringBitmap childTids = directRefine(child, parentTids, currentTuple);
            if (childTids.isEmpty()) {
                continue;
            }

            visitAfterPredicate(
                    child,
                    childTids,
                    currentTuple,
                    currentNullCols,
                    strategy,
                    votes);
        }
    }

    private void visitAfterPredicate(
            TrieNode node,
            RoaringBitmap tids,
            int currentTuple,
            Set<Integer> currentNullCols,
            VotingStrategy strategy,
            VoteState votes) {

        if (node.terminalDCCount > 0) {
            // CURRENT exactly follows production MissDC: support is counted only
            // at refinement-tree leaves. TUPLE/DC use explicit DC terminals.
            if (strategy != VotingStrategy.CURRENT || node.children.isEmpty()) {
                votes.observe(tids, node.terminalDCCount);
            }
        }

        visitChildren(node, tids, currentTuple, currentNullCols, strategy, votes);
    }

    private boolean predicateUnavailable(TrieNode node, Set<Integer> currentNullCols) {
        int predicateColumn = node.refiner.predicate.getPredicate().getCol1().ColumnIndex;
        return currentNullCols.contains(predicateColumn);
    }

    /**
     * Equivalent to Refiner.refine(), but returns a fresh bitmap obtained from
     * direct RoaringBitmap operations instead of cloning parentTids first.
     */
    private RoaringBitmap directRefine(TrieNode node, RoaringBitmap parentTids, int tid) {
        IndexedPredicate predicate = node.refiner.predicate;
        Column col = node.refiner.col;
        float valueForCol = predicate.getPredicate().getCol1().getValueAt(tid);

        RoaringBitmap result;

        if (predicate.isUNEQ()) {
            RoaringBitmap equal = predicate
                    .getInverse()
                    .getEqualityIndex()
                    .geTidsOfEqualValues(valueForCol);
            result = equal == null
                    ? parentTids.clone()
                    : RoaringBitmap.andNot(parentTids, equal);

        } else if (predicate.isEQ()) {
            RoaringBitmap equal = predicate
                    .getEqualityIndex()
                    .geTidsOfEqualValues(valueForCol);
            result = equal == null
                    ? new RoaringBitmap()
                    : RoaringBitmap.and(parentTids, equal);

        } else if (predicate.isLT()) {
            RoaringBitmap lt = getLTBitmap(predicate, col, valueForCol);
            result = lt == null
                    ? new RoaringBitmap()
                    : RoaringBitmap.and(parentTids, lt);

        } else if (predicate.isLTE()) {
//             Preserve the exact current Refiner sequence.
            NumericalColumnPredicateGroup group = numericalGroup(col);
            RoaringBitmap lt = getLTBitmap(group.getLt(), col, valueForCol);
            if (lt == null) {
                result = new RoaringBitmap();
            } else {
                result = RoaringBitmap.and(parentTids, lt);
                RoaringBitmap equal = group.getEq()
                        .getEqualityIndex()
                        .geTidsOfEqualValues(valueForCol);
                if (equal == null) {
                    result.clear();
                } else {
                    result.and(equal);
                }
            }
//            NumericalColumnPredicateGroup group = numericalGroup(col);
//
//            RoaringBitmap lt = getLTBitmap(group.getLt(), col, valueForCol);
//
//            RoaringBitmap equal = group.getEq()
//                    .getEqualityIndex()
//                    .geTidsOfEqualValues(valueForCol);
//
//            RoaringBitmap lte = lt == null
//                    ? new RoaringBitmap()
//                    : lt.clone();
//
//            if (equal != null) {
//                lte.or(equal);
//            }
//
//            result = RoaringBitmap.and(parentTids, lte);

        } else if (predicate.isGT()) {
            NumericalColumnPredicateGroup group = numericalGroup(col);
            RoaringBitmap lt = getLTBitmap(group.getLt(), col, valueForCol);
            result = lt == null
                    ? parentTids.clone()
                    : RoaringBitmap.andNot(parentTids, lt);

            RoaringBitmap equal = group.getEq()
                    .getEqualityIndex()
                    .geTidsOfEqualValues(valueForCol);
            if (equal != null) {
                result.andNot(equal);
            }

        } else if (predicate.isGTE()) {
            NumericalColumnPredicateGroup group = numericalGroup(col);
            RoaringBitmap lt = getLTBitmap(group.getLt(), col, valueForCol);
            result = lt == null
                    ? parentTids.clone()
                    : RoaringBitmap.andNot(parentTids, lt);

        } else {
            // Defensive fallback for future predicate types.
            result = parentTids.clone();
            node.refiner.refine(predicateSpace, result, tid);
            return result;
        }

        RoaringBitmap colMissing = col.getBitmapOfMissingTids();
        if (colMissing != null) {
            result.andNot(colMissing);
        }
        return result;
    }

    private RoaringBitmap getLTBitmap(IndexedPredicate predicate, Column col, float value) {
        NumericalColumnPredicateGroup group = numericalGroup(col);
        return group.isBinnedLT()
                ? predicate.getLtBinnedIndex().getTidsLTValues(value)
                : predicate.getLtIndex().getTidsLTValues(value);
    }

    private NumericalColumnPredicateGroup numericalGroup(Column col) {
        return (NumericalColumnPredicateGroup) predicateSpace.getColumn2GroupMap().get(col);
    }

    private static final class VoteState {
        private final Column targetColumn;
        private final VotingStrategy strategy;
        private final Map<Float, Integer> support = new HashMap<>();
        private final RoaringBitmap tupleVoters = new RoaringBitmap();
        private long totalVotes;

        VoteState(Column targetColumn, VotingStrategy strategy) {
            this.targetColumn = targetColumn;
            this.strategy = strategy;
        }

        void observe(RoaringBitmap donorTids, int dcCount) {
            if (donorTids == null || donorTids.isEmpty() || dcCount <= 0) {
                return;
            }

            switch (strategy) {
                case CURRENT -> observeCurrent(donorTids, dcCount);
                case TUPLE -> observeTuple(donorTids);
                case DC -> observeDC(donorTids, dcCount);
            }
        }

        private void observeCurrent(RoaringBitmap donorTids, int dcCount) {
            for (int donorTid : donorTids) {
                float candidate = targetColumn.getValueAt(donorTid);
                support.merge(candidate, dcCount, Integer::sum);
                totalVotes += dcCount;
            }
        }

        private void observeTuple(RoaringBitmap donorTids) {
            RoaringBitmap newVoters = RoaringBitmap.andNot(donorTids, tupleVoters);
            if (newVoters.isEmpty()) {
                return;
            }

            for (int donorTid : newVoters) {
                float candidate = targetColumn.getValueAt(donorTid);
                support.merge(candidate, 1, Integer::sum);
                totalVotes++;
            }
            tupleVoters.or(newVoters);
        }

        private void observeDC(RoaringBitmap donorTids, int dcCount) {
            Map<Float, Integer> localSupport = new HashMap<>();
            for (int donorTid : donorTids) {
                float candidate = targetColumn.getValueAt(donorTid);
                localSupport.merge(candidate, 1, Integer::sum);
            }

            LocalWinner localWinner = localWinner(localSupport);
            if (localWinner.candidate == null || localWinner.tie) {
                return;
            }

            support.merge(localWinner.candidate, dcCount, Integer::sum);
            totalVotes += dcCount;
        }

        Selection finish() {
            if (support.isEmpty()) {
                return new Selection(null, 0L, totalVotes);
            }

            // Match production MissDC's winner selection for CURRENT, including
            // its HashMap tie behavior. Use the same rule for the other modes so
            // only the voting semantics change.
            Map.Entry<Float, Integer> winner = support.entrySet()
                    .stream()
                    .max(Map.Entry.comparingByValue())
                    .orElseThrow();

            return new Selection(winner.getKey(), winner.getValue(), totalVotes);
        }

        private static LocalWinner localWinner(Map<Float, Integer> support) {
            if (support.isEmpty()) {
                return new LocalWinner(null, false);
            }

            int max = Integer.MIN_VALUE;
            Float candidate = null;
            boolean tie = false;

            for (Map.Entry<Float, Integer> entry : support.entrySet()) {
                int value = entry.getValue();
                if (value > max) {
                    max = value;
                    candidate = entry.getKey();
                    tie = false;
                } else if (value == max
                        && candidate != null
                        && Float.compare(candidate, entry.getKey()) != 0) {
                    tie = true;
                }
            }

            return new LocalWinner(candidate, tie);
        }

        private record LocalWinner(Float candidate, boolean tie) {}
    }
}