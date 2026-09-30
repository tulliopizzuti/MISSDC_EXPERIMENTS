package org.missdc.discovery.enumeration;


import org.missdc.dc.predicates.IndexedPredicate;
import org.missdc.dc.predicates.groups.ColumnPredicateGroup;
import org.missdc.dc.predicates.space.PredicateSpace;
import org.missdc.discovery.evidence.EvidenceSet;
import org.missdc.utils.bitset.BitUtils;
import org.missdc.utils.bitset.IBitSet;
import org.missdc.utils.bitset.LongBitSet;
import org.missdc.utils.misc.Object2IndexMapper;
import org.missdc.utils.search.NTreeSearch;
import org.missdc.utils.search.TreeSearch;
import it.unimi.dsi.fastutil.objects.Object2LongOpenHashMap;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;

public class HybridEvidenceInversion implements DCEnumeration {

	private PredicateSpace pSpace;
	private Object2LongOpenHashMap<BitSet> eviSet;
	private Map<Integer, Set<Integer>> pred2PredGroupMap;
	private Set<BitSet> covers;

	private BitSet initialEviIDs;
	private Object2IndexMapper<BitSet> eviIndex; // each evi has an ID
	private Map<Integer, BitSet> pred2EviMap;// for each predicate p, the eviIDs of evidence containing p

	private boolean checkMinimals;

	private boolean parallel;

	private final int maxDCSize;

	public static final int UNBOUNDED = EvidenceInversion.UNBOUNDED;

	public HybridEvidenceInversion(PredicateSpace predicateSpace, EvidenceSet evidenceSet) {
		this(predicateSpace, evidenceSet, true, true, UNBOUNDED);
	}

	public HybridEvidenceInversion(PredicateSpace predicateSpace, EvidenceSet evidenceSet, int maxDCSize) {
		this(predicateSpace, evidenceSet, true, true, maxDCSize);
	}

	public HybridEvidenceInversion(PredicateSpace predicateSpace, EvidenceSet evidenceSet, boolean parallel) {
		this(predicateSpace, evidenceSet, true, parallel, UNBOUNDED);
	}

	public HybridEvidenceInversion(PredicateSpace predicateSpace, EvidenceSet evidenceSet, boolean checkMinimals,
			boolean parallel) {
		this(predicateSpace, evidenceSet, checkMinimals, parallel, UNBOUNDED);
	}

	public HybridEvidenceInversion(PredicateSpace predicateSpace, EvidenceSet evidenceSet, boolean checkMinimals,
			boolean parallel, int maxDCSize) {
		this.pSpace = predicateSpace;
		this.eviSet = evidenceSet.getEvidence2CountMap();
		if (maxDCSize != UNBOUNDED && maxDCSize < 1) {
			throw new IllegalArgumentException("maxDCSize must be >= 1 or UNBOUNDED (-1)");
		}

		this.checkMinimals = checkMinimals;
		this.parallel = parallel;
		this.maxDCSize = maxDCSize;
		this.covers = new HashSet<>();

		pred2PredGroupMap = new HashMap<>();
		for (ColumnPredicateGroup pGroup : pSpace.getPredicateGroups()) {
			Set<Integer> pids = new HashSet<>();

			for (int i = pGroup.getPredicateIDs().nextSetBit(0); i != -1; i = pGroup.getPredicateIDs()
					.nextSetBit(i + 1)) {
				pids.add(i);
			}

			for (int i = pGroup.getPredicateIDs().nextSetBit(0); i != -1; i = pGroup.getPredicateIDs()
					.nextSetBit(i + 1)) {
				pred2PredGroupMap.put(i, pids);

			}
		}

	}

	public HybridEvidenceInversion(HybridEvidenceInversion base) {

		this.pSpace = base.pSpace;
		this.eviSet = base.eviSet;
		this.pred2EviMap = base.pred2EviMap;
		this.eviIndex = base.eviIndex;
		this.pred2PredGroupMap = base.pred2PredGroupMap;
		this.checkMinimals = base.checkMinimals;
		this.parallel = base.parallel;
		this.maxDCSize = base.maxDCSize;

		this.covers = new HashSet<>();
	}

	public Set<BitSet> searchDCs() {

		buildEvidenceIndexes();

		List<Integer> initialSortedPreds = new ArrayList<>();
		List<Integer> predIdSequence = new ArrayList<>();

		for (int pid = 0; pid < pSpace.size(); pid++) {
			initialSortedPreds.add(pid);
			predIdSequence.add(pid);
		}

		sortInitialPredicates(initialSortedPreds, initialEviIDs);

		Set<BitSet> allCovers = ConcurrentHashMap.newKeySet();

		Stream<Integer> stream;

		if (parallel)
			stream = predIdSequence.parallelStream();

		else
			stream = predIdSequence.stream();

		stream.forEach(i -> {

			int pid = initialSortedPreds.get(i);

			if (pred2EviMap.get(pid).cardinality() == 0) {
				BitSet dc = new BitSet();
				dc.set(pid);
				allCovers.add(dc);
				return;
			}

			List<Integer> currPreds = new ArrayList<>(initialSortedPreds);
			currPreds = currPreds.subList(i + 1, currPreds.size());
			currPreds.removeAll(pred2PredGroupMap.get(pid));

			Set<Integer> predicatets2Remove = new HashSet<>();

			for (int j = 0; j < i; j++) {
				predicatets2Remove.add(initialSortedPreds.get(j));
			}

			BitSet currEvis = pred2EviMap.get(pid);

			int maxPartialSize = maxDCSize == UNBOUNDED
					? EvidenceInversion.UNBOUNDED
					: maxDCSize - 1;

			EvidenceInversion search = new EvidenceInversion(pSpace,
					getModuloEvi(currEvis, pSpace.getPredicateById(pid), predicatets2Remove),
					pSpace.getPredicateById(pid), predicatets2Remove, maxPartialSize);

			Set<BitSet> partialDCs = search.searchDCs();

			for (BitSet positiveCover : partialDCs) {

				BitSet dc = positiveCover;

				dc.set(pid);

				if (maxDCSize == UNBOUNDED || dc.cardinality() <= maxDCSize) {
					allCovers.add(dc);
				}

			}

		});

		covers = allCovers;

		if (checkMinimals)
			checkMinimality();

		return covers;
	}

	private Set<BitSet> getModuloEvi(BitSet currEvis, IndexedPredicate indexedPredicate, Set<Integer> preds2Remove) {

		Set<BitSet> moduloEvi = new HashSet<>();

		for (int eviID = currEvis.nextSetBit(0); eviID >= 0; eviID = currEvis.nextSetBit(eviID + 1)) {

			BitSet bs = (BitSet) eviIndex.getObject(eviID).clone();

			pred2PredGroupMap.get(indexedPredicate.getPredicateId()).forEach(i -> bs.clear(i));

			preds2Remove.forEach(p -> bs.clear(p));

			moduloEvi.add(bs);

		}

		return moduloEvi;
	}

	private void sortInitialPredicates(List<Integer> currPreds, BitSet currEvis) {

		long predCounts[] = new long[pSpace.size()];

		// get counts for each evidence
		for (int pid : currPreds) {
			BitSet andResul = BitUtils.getAndBitSet(currEvis, pred2EviMap.get(pid));

			predCounts[pid] = andResul.cardinality();
		}

		currPreds.sort(new Comparator<Integer>() {
			@Override
			public int compare(Integer o1, Integer o2) {
				return Long.compare(predCounts[o1], predCounts[o2]);// ascending, we want the least frequent first
			}
		});

	}

	private BitSet filterCoveredEvidences(BitSet evis, int pid, List<Integer> newCurrPreds) {

		BitSet filteredEvis = (BitSet) evis.clone();
		filteredEvis.and(pred2EviMap.get(pid));

		BitSet preds = new BitSet();
		newCurrPreds.forEach(i -> preds.set(i));

		for (int eviID = filteredEvis.nextSetBit(0); eviID >= 0; eviID = filteredEvis.nextSetBit(eviID + 1)) {
			// the remaining predicates cannot form a DC as they are all contained in an
			// evidence
			if (BitUtils.isSubSetOf(preds, eviIndex.getObject(eviID))) {
				return null;
			}
		}

		return filteredEvis;
	}

	private void buildEvidenceIndexes() {

		pred2EviMap = new HashMap<>();

		for (int pid = 0; pid < pSpace.size(); pid++) {
			pred2EviMap.put(pid, new BitSet());
		}

		initialEviIDs = new BitSet();

		eviIndex = new Object2IndexMapper<BitSet>();

		List<BitSet> eviList = minimize(eviSet.keySet());

		for (BitSet evi : eviList) {

			int eviID = eviIndex.getIndex(evi);

			initialEviIDs.set(eviID);

			for (int pID = evi.nextSetBit(0); pID >= 0; pID = evi.nextSetBit(pID + 1)) {

				pred2EviMap.get(pID).set(eviID);
			}

		}

	}

	private List<BitSet> minimize(final Set<BitSet> evis) {

		List<IBitSet> sortedNegCover = new ArrayList<>();
		evis.forEach(evi -> sortedNegCover.add(LongBitSet.FACTORY.create(evi)));

		Collections.sort(sortedNegCover, new Comparator<IBitSet>() {
			@Override
			public int compare(IBitSet o1, IBitSet o2) {
				int erg = Integer.compare(o2.cardinality(), o1.cardinality());
				return erg != 0 ? erg : o2.compareTo(o1);
			}
		});

		TreeSearch neg = new TreeSearch();
		sortedNegCover.stream().forEach(invalid -> addInvalidToNeg(neg, invalid));

		final ArrayList<BitSet> list = new ArrayList<>();

		neg.forEach(invalidFD -> list.add(invalidFD.toBitSet()));

		return list;
	}

	private void addInvalidToNeg(TreeSearch neg, IBitSet invalid) {
		if (neg.findSuperSet(invalid) != null)
			return;

		neg.getAndRemoveGeneralizations(invalid);
		neg.add(invalid);
	}

	private void checkMinimality() {

		NTreeSearch nt = new NTreeSearch();
		for (BitSet key : covers) {
			nt.add(LongBitSet.FACTORY.create(key));
		}

		Set<BitSet> nonGeneralized = new HashSet<>();

		for (BitSet key : covers) {

			IBitSet bs = LongBitSet.FACTORY.create(key);

			nt.remove(bs);

			if (!nt.containsSubset(bs)) {
				nonGeneralized.add(key);
			}

			nt.add(bs);

		}
		covers = nonGeneralized;

	}

}