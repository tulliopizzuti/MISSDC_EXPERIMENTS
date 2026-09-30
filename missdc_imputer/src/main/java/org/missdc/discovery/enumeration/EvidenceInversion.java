package org.missdc.discovery.enumeration;

import org.missdc.dc.predicates.IndexedPredicate;
import org.missdc.dc.predicates.groups.ColumnPredicateGroup;
import org.missdc.dc.predicates.space.PredicateSpace;
import org.missdc.discovery.evidence.EvidenceSet;
import org.missdc.utils.bitset.IBitSet;
import org.missdc.utils.bitset.LongBitSet;
import org.missdc.utils.search.ITreeSearch;
import org.missdc.utils.search.TranslatingTreeSearch;
import org.missdc.utils.search.TreeSearch;

import java.util.*;

public class EvidenceInversion implements DCEnumeration {

	public static final int UNBOUNDED = TranslatingTreeSearch.UNBOUNDED;

	private PredicateSpace pSpace;
	private Set<BitSet> evidenceBitSets;
	private List<IBitSet> bitsetList = new ArrayList<>();
	private final Collection<IBitSet> startBitsets;
	private TranslatingTreeSearch posCover;
	private final int maxCoverSize;

	public EvidenceInversion(PredicateSpace pSpace, EvidenceSet eviSet) {
		this(pSpace, eviSet, UNBOUNDED);
	}

	public EvidenceInversion(PredicateSpace pSpace, EvidenceSet eviSet, int maxCoverSize) {
		this(pSpace, eviSet, (TranslatingTreeSearch) null, maxCoverSize);
		this.startBitsets.add(LongBitSet.FACTORY.create());
	}

	private EvidenceInversion(PredicateSpace pSpace, EvidenceSet eviSet, TranslatingTreeSearch tree, int maxCoverSize) {

		this.pSpace = pSpace;

		this.evidenceBitSets = eviSet.getEvidence2CountMap().keySet();

		this.startBitsets = new ArrayList<IBitSet>();
		this.posCover = tree;
		this.maxCoverSize = validateMaxCoverSize(maxCoverSize);

		for (ColumnPredicateGroup group : pSpace.getPredicateGroups()) {

			IBitSet bitset = LongBitSet.FACTORY.create(group.getPredicateIDs());

			bitsetList.add(bitset);

		}

	}

	/* for modulo approach */
	public EvidenceInversion(PredicateSpace pSpace, Set<BitSet> evidenceBitSets, IndexedPredicate p,
			Set<Integer> predicatets2Remove) {
		this(pSpace, evidenceBitSets, p, predicatets2Remove, UNBOUNDED);
	}

	public EvidenceInversion(PredicateSpace pSpace, Set<BitSet> evidenceBitSets, IndexedPredicate p,
			Set<Integer> predicatets2Remove, int maxCoverSize) {
		this(pSpace, evidenceBitSets, (TranslatingTreeSearch) null, p, predicatets2Remove, maxCoverSize);
		this.startBitsets.add(LongBitSet.FACTORY.create());
	}

	private EvidenceInversion(PredicateSpace pSpace, Set<BitSet> evidenceBitSets, TranslatingTreeSearch tree,
			IndexedPredicate p, Set<Integer> preds2Remove, int maxCoverSize) {

		this.pSpace = pSpace;
		this.evidenceBitSets = evidenceBitSets;

		this.startBitsets = new ArrayList<IBitSet>();
		this.posCover = tree;
		this.maxCoverSize = validateMaxCoverSize(maxCoverSize);

		for (ColumnPredicateGroup group : pSpace.getPredicateGroups()) {

			if (group.getPredicates().contains(p))
				continue;

			IBitSet bitset = LongBitSet.FACTORY.create(group.getPredicateIDs());

			preds2Remove.forEach(i -> {
				if (bitset.get(i))
					bitset.clear(i);
			});

			bitsetList.add(bitset);

		}

	}

	/* for hybrid approach */
	public EvidenceInversion(PredicateSpace pSpace, Set<BitSet> evidenceBitSets, Set<Integer> predicatets2Remove) {
		this(pSpace, evidenceBitSets, predicatets2Remove, UNBOUNDED);
	}

	public EvidenceInversion(PredicateSpace pSpace, Set<BitSet> evidenceBitSets, Set<Integer> predicatets2Remove,
			int maxCoverSize) {
		this(pSpace, evidenceBitSets, (TranslatingTreeSearch) null, predicatets2Remove, maxCoverSize);
		this.startBitsets.add(LongBitSet.FACTORY.create());

	}

	private EvidenceInversion(PredicateSpace pSpace, Set<BitSet> evidenceBitSets, TranslatingTreeSearch tree,
			Set<Integer> preds2Remove, int maxCoverSize) {

		this.pSpace = pSpace;
		this.evidenceBitSets = evidenceBitSets;
		this.startBitsets = new ArrayList<IBitSet>();
		this.posCover = tree;
		this.maxCoverSize = validateMaxCoverSize(maxCoverSize);

		for (ColumnPredicateGroup group : pSpace.getPredicateGroups()) {

			IBitSet bitset = LongBitSet.FACTORY.create(group.getPredicateIDs());

			preds2Remove.forEach(i -> {
				if (bitset.get(i))
					bitset.clear(i);
			});
			bitsetList.add(bitset);
		}

	}

	public Set<BitSet> searchDCs() {

		Set<BitSet> dcs = new HashSet<>();
		for (IBitSet bs : getBitsets()) {
			dcs.add(bs.toBitSet());
		}

		return dcs;

	}

	private Collection<IBitSet> getBitsets() {

		if (posCover == null) {
			int[] counts = getCounts();
			posCover = new TranslatingTreeSearch(counts, bitsetList, maxCoverSize);
		}

		List<IBitSet> sortedNegCover = new ArrayList<IBitSet>();

		for (BitSet bitset : evidenceBitSets) {
			sortedNegCover.add(LongBitSet.FACTORY.create(bitset)); // from BitSet into IBitset
		}

		sortedNegCover = minimize(sortedNegCover);

		mostGeneralDCs(posCover);

		Collections.sort(sortedNegCover, posCover.getComparator());

		for (int i = 0; i < sortedNegCover.size(); ++i) {
			posCover.handleInvalid(sortedNegCover.get(i));
		}

		Collection<IBitSet> result = new ArrayList<IBitSet>();
		posCover.forEach(bs -> result.add(bs));

		return result;

	}

	private void mostGeneralDCs(ITreeSearch posCover) {
		for (IBitSet start : startBitsets) {
			posCover.add(start);
		}
	}

	private List<IBitSet> minimize(final List<IBitSet> sortedNegCover) {

		Collections.sort(sortedNegCover, new Comparator<IBitSet>() {
			@Override
			public int compare(IBitSet o1, IBitSet o2) {
				int erg = Integer.compare(o2.cardinality(), o1.cardinality());
				return erg != 0 ? erg : o2.compareTo(o1);
			}
		});

		TreeSearch neg = new TreeSearch();
		sortedNegCover.stream().forEach(invalid -> addInvalidToNeg(neg, invalid));

		final ArrayList<IBitSet> list = new ArrayList<IBitSet>();
		neg.forEach(invalidFD -> list.add(invalidFD));
		return list;
	}

	private void addInvalidToNeg(TreeSearch neg, IBitSet invalid) {
		if (neg.findSuperSet(invalid) != null)
			return;

		neg.getAndRemoveGeneralizations(invalid);
		neg.add(invalid);
	}

	private static int validateMaxCoverSize(int maxCoverSize) {
		// -1 is the only unbounded sentinel. Zero is valid internally because
		// HybridEvidenceInversion searches a partial cover of size k-1 and then
		// adds its fixed starting predicate. For example, final k=1 => partial 0.
		if (maxCoverSize != UNBOUNDED && maxCoverSize < 0) {
			throw new IllegalArgumentException(
					"maxCoverSize must be >= 0 or UNBOUNDED (-1)");
		}
		return maxCoverSize;
	}

	private int[] getCounts() {
		int[] counts = new int[pSpace.size()];

		for (BitSet bitset : evidenceBitSets) {

			for (int i = bitset.nextSetBit(0); i >= 0; i = bitset.nextSetBit(i + 1)) {
				counts[i]++;
			}
		}
		return counts;
	}

}