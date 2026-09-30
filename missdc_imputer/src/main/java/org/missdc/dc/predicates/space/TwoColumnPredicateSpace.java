package org.missdc.dc.predicates.space;

import org.missdc.dc.predicates.IndexedPredicate;
import org.missdc.dc.predicates.Predicate;
import org.missdc.dc.predicates.groups.ColumnPredicateGroup;
import org.missdc.dc.predicates.groups.twocol.TwoColumnNumericalPredicateGroup;
import org.missdc.dc.predicates.groups.twocol.TwoColumnPredicateGroup;
import org.missdc.utils.misc.Object2IndexMapper;

import java.util.*;

public class TwoColumnPredicateSpace {

	protected PredicateSpace oneColspace;

	protected Object2IndexMapper<IndexedPredicate> predicateIndex;

	protected Set<TwoColumnPredicateGroup> twoColCatpredGroups;
	protected Set<TwoColumnPredicateGroup> twoColSingleTupleCatPredGroups;
	protected Set<TwoColumnPredicateGroup> twoColTwoTupleCatPredGroups;

	protected Set<TwoColumnNumericalPredicateGroup> twoColNumericalpredGroups;
	protected Set<TwoColumnNumericalPredicateGroup> twoColSingleTupleNumericalPredGroups;
	protected Set<TwoColumnNumericalPredicateGroup> twoColTwoTupleNumericalPredGroups;

	public TwoColumnPredicateSpace(PredicateSpace oneColspace) {

		this.oneColspace = oneColspace;
		this.predicateIndex = oneColspace.predicateIndex;

		this.twoColCatpredGroups = new LinkedHashSet<>();

		this.twoColSingleTupleCatPredGroups = new LinkedHashSet<>();

		this.twoColTwoTupleCatPredGroups = new LinkedHashSet<>();

		this.twoColNumericalpredGroups = new LinkedHashSet<>();

		this.twoColSingleTupleNumericalPredGroups = new LinkedHashSet<>();

		this.twoColTwoTupleNumericalPredGroups = new LinkedHashSet<>();

	}

	public PredicateSpace getOneColspace() {
		return oneColspace;
	}

	public Object2IndexMapper<IndexedPredicate> getPredicateIndex() {
		return predicateIndex;
	}

	public void addTwoColSingleTupleCategoricalPredicateGroup(
			TwoColumnPredicateGroup twoColSingleTupleCategoricalColumnGroup) {

		twoColCatpredGroups.add(twoColSingleTupleCategoricalColumnGroup);
		twoColSingleTupleCatPredGroups.add(twoColSingleTupleCategoricalColumnGroup);

	}

	public void addTwoColTwoTupleCategoricalPredicateGroup(
			TwoColumnPredicateGroup twoColTwoTupleCategoricalColumnGroup) {

		twoColCatpredGroups.add(twoColTwoTupleCategoricalColumnGroup);
		twoColTwoTupleCatPredGroups.add(twoColTwoTupleCategoricalColumnGroup);

	}

	public void addTwoColSingleTupleNumericalPredicateGroup(
			TwoColumnNumericalPredicateGroup twoColSingleTupleNumericalColumnGroup) {

		twoColNumericalpredGroups.add(twoColSingleTupleNumericalColumnGroup);
		twoColSingleTupleNumericalPredGroups.add(twoColSingleTupleNumericalColumnGroup);

	}

	public void addTwoColTwoTupleNumericalPredicateGroup(
			TwoColumnNumericalPredicateGroup twoColTwoTupleNumericalColumnGroup) {

		twoColNumericalpredGroups.add(twoColTwoTupleNumericalColumnGroup);
		twoColTwoTupleNumericalPredGroups.add(twoColTwoTupleNumericalColumnGroup);

	}

	public Set<TwoColumnPredicateGroup> getTwoColCatpredGroups() {
		return twoColCatpredGroups;
	}

	public Set<TwoColumnNumericalPredicateGroup> getTwoColNumericalpredGroups() {
		return twoColNumericalpredGroups;
	}

	public int size() {

		return predicateIndex.size();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(oneColspace);

		sb.append("\n");
		for (ColumnPredicateGroup group : twoColCatpredGroups) {

			sb.append(group);
			sb.append("\n");

		}

		for (ColumnPredicateGroup group : twoColNumericalpredGroups) {

			sb.append(group);
			sb.append("\n");

		}

		return sb.toString();
	}

	public List<String> convertIntoPredicateStr(BitSet bitSet) {

		List<String> strSet = new ArrayList<String>();

		for (int i = bitSet.nextSetBit(0); i != -1; i = bitSet.nextSetBit(i + 1)) {
			Predicate p = predicateIndex.getObject(i).getPredicate();
			strSet.add("(" + i + ")" + p.toString().toLowerCase());
		}
		return strSet;
	}

	public Set<TwoColumnPredicateGroup> getTwoColSingleTupleCatPredGroups() {
		return twoColSingleTupleCatPredGroups;
	}

	public Set<TwoColumnNumericalPredicateGroup> getTwoColSingleTupleNumericalPredGroups() {
		return twoColSingleTupleNumericalPredGroups;
	}

	public Set<TwoColumnPredicateGroup> getTwoColTwoTupleCatPredGroups() {
		return twoColTwoTupleCatPredGroups;
	}

	public Set<TwoColumnNumericalPredicateGroup> getTwoColTwoTupleNumericalPredGroups() {
		return twoColTwoTupleNumericalPredGroups;
	}

	public String toDCstring(BitSet dc) {

		List<String> preds = new ArrayList<String>();

		for (int i = dc.nextSetBit(0); i != -1; i = dc.nextSetBit(i + 1)) {
			preds.add(predicateIndex.getObject(i).getPredicate().toString());

		}

		String strDC = "not(" + String.join(" and ", preds).toLowerCase() + ")";

		return strDC;

	}

	public IndexedPredicate getPredicateById(int pid) {
		
		return predicateIndex.getObject(pid);
	}

}