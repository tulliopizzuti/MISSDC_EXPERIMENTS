package org.missdc.dc.predicates.space.builder;

import org.missdc.dc.predicates.IndexedPredicate;
import org.missdc.dc.predicates.RelationalOperator;
import org.missdc.dc.predicates.TuplePairSingleColumnPredicate;
import org.missdc.dc.predicates.groups.ColumnPredicateGroup;
import org.missdc.dc.predicates.groups.NumericalColumnPredicateGroup;
import org.missdc.dc.predicates.space.PredicateProvider;
import org.missdc.dc.predicates.space.PredicateSpace;
import org.missdc.input.Table;
import org.missdc.input.columns.CategoricalColumn;
import org.missdc.input.columns.Column;
import org.missdc.utils.misc.Object2IndexMapper;

public class PredicateSpaceBuilder {

	public PredicateSpace build(Table input) {

		Object2IndexMapper<IndexedPredicate> predicateIndex = new Object2IndexMapper<>();

		PredicateSpace space = new PredicateSpace(predicateIndex);

		for (Column column : input.getAllColumns().values()) {

			if (column instanceof CategoricalColumn c) {

				space.addCategoricalPredicateGroup(createCategoricalColumnGroup(space, column));

			} else { // NumericalColumn

				space.addNumericalPredicateGroup(createSingleNumericalColumnGroup(space, column));

			}

		}

		PredicateProvider.initStaticInstance(space);

		return space;
	}

	private ColumnPredicateGroup createCategoricalColumnGroup(PredicateSpace space, Column column) {

		Object2IndexMapper<IndexedPredicate> predicateIndex = space.getPredicateIndex();

		IndexedPredicate eq = new IndexedPredicate(space,
				new TuplePairSingleColumnPredicate(column, RelationalOperator.EQUAL), false);
		IndexedPredicate uneq = new IndexedPredicate(space,
				new TuplePairSingleColumnPredicate(column, RelationalOperator.UNEQUAL), false);

		int eqIdx = predicateIndex.getIndex(eq);
		int uneqIdx = predicateIndex.getIndex(uneq);

		eq.setPredicateId(eqIdx);
		eq.setPredicateInverseId(uneqIdx);
		eq.setImplicationIdSet(eqIdx);
		eq.setPredicateSymmetricId(eqIdx);

		uneq.setPredicateId(uneqIdx);
		uneq.setPredicateInverseId(eqIdx);
		uneq.setImplicationIdSet(uneqIdx);
		uneq.setPredicateSymmetricId(uneqIdx);

		eq.buildCategoricalMask(eqIdx, uneqIdx);

		ColumnPredicateGroup columnPredicateGroup = new ColumnPredicateGroup(eq, uneq);

		return columnPredicateGroup;

	}

	private NumericalColumnPredicateGroup createSingleNumericalColumnGroup(PredicateSpace space, Column column) {

		Object2IndexMapper<IndexedPredicate> predicateIndex = space.getPredicateIndex();

		IndexedPredicate eq = new IndexedPredicate(space,
				new TuplePairSingleColumnPredicate(column, RelationalOperator.EQUAL), true);

		IndexedPredicate uneq = new IndexedPredicate(space,
				new TuplePairSingleColumnPredicate(column, RelationalOperator.UNEQUAL), true);

		IndexedPredicate lt = new IndexedPredicate(space,
				new TuplePairSingleColumnPredicate(column, RelationalOperator.LESS), true);

		IndexedPredicate lte = new IndexedPredicate(space,
				new TuplePairSingleColumnPredicate(column, RelationalOperator.LESS_EQUAL), true);

		IndexedPredicate gt = new IndexedPredicate(space,
				new TuplePairSingleColumnPredicate(column, RelationalOperator.GREATER), true);

		IndexedPredicate gte = new IndexedPredicate(space,
				new TuplePairSingleColumnPredicate(column, RelationalOperator.GREATER_EQUAL), true);

		int eqIdx = predicateIndex.getIndex(eq);
		int uneqIdx = predicateIndex.getIndex(uneq);
		int ltIdx = predicateIndex.getIndex(lt);
		int lteIdx = predicateIndex.getIndex(lte);
		int gtIdx = predicateIndex.getIndex(gt);
		int gteIdx = predicateIndex.getIndex(gte);

		// for negation and implication, see Table 2 of Xu Chu's paper on dc discovery

		eq.setPredicateId(eqIdx);
		eq.setPredicateInverseId(uneqIdx);
		eq.setImplicationIdSet(eqIdx, gteIdx, lteIdx);
		eq.setPredicateSymmetricId(eqIdx);

		uneq.setPredicateId(uneqIdx);
		uneq.setPredicateInverseId(eqIdx);
		uneq.setImplicationIdSet(uneqIdx);
		uneq.setPredicateSymmetricId(uneqIdx);

		lt.setPredicateId(ltIdx);
		lt.setPredicateInverseId(gteIdx);
		lt.setImplicationIdSet(ltIdx, lteIdx, uneqIdx);
		lt.setPredicateSymmetricId(gtIdx);

		lte.setPredicateId(lteIdx);
		lte.setPredicateInverseId(gtIdx);
		lte.setImplicationIdSet(lteIdx);
		lte.setPredicateSymmetricId(gteIdx);

		gt.setPredicateId(gtIdx);
		gt.setPredicateInverseId(lteIdx);
		gt.setImplicationIdSet(gtIdx, gteIdx, uneqIdx);
		gt.setPredicateSymmetricId(ltIdx);

		gte.setPredicateId(gteIdx);
		gte.setPredicateInverseId(ltIdx);
		gte.setImplicationIdSet(gteIdx);
		gte.setPredicateSymmetricId(lteIdx);

		eq.buildNumericalEqRangeMask(eqIdx, uneqIdx, ltIdx, lteIdx, gtIdx, gteIdx);

		lt.buildNumericalRangeRangeMask(ltIdx, lteIdx, gtIdx, gteIdx);

		gt.buildNumericalRangeRangeMask(ltIdx, lteIdx, gtIdx, gteIdx);

		NumericalColumnPredicateGroup columnPredicateGroup = new NumericalColumnPredicateGroup(eq, uneq, lt, lte, gt,
				gte);

		return columnPredicateGroup;

	}

}