package org.missdc.imputation;


import org.missdc.dc.predicates.IndexedPredicate;
import org.missdc.dc.predicates.groups.NumericalColumnPredicateGroup;
import org.missdc.dc.predicates.space.PredicateSpace;
import org.missdc.input.columns.Column;
import org.roaringbitmap.RoaringBitmap;

import java.util.Objects;

public class Refiner {

    IndexedPredicate predicate;
    Column col;


    public Refiner(IndexedPredicate predicate) {
        this.predicate = predicate;
        if (predicate != null)
            this.col = predicate.getPredicate().getCol1();
    }

    protected void refine(PredicateSpace predicateicateSpace, RoaringBitmap tids, int tid) {

        RoaringBitmap colMissing = col.getBitmapOfMissingTids();

        if (colMissing != null)
            tids.andNot(colMissing);

        if (predicate.isUNEQ()) {

            float valueForCol = predicate.getPredicate().getCol1().getValueAt(tid);
            RoaringBitmap equalC2 = predicate.getInverse().getEqualityIndex().geTidsOfEqualValues(valueForCol);
            tids.andNot(equalC2); // remove tids that are equal to the value

        } else if (predicate.isEQ()) {

            float valueForCol = predicate.getPredicate().getCol1().getValueAt(tid);
            RoaringBitmap equalC2 = predicate.getEqualityIndex().geTidsOfEqualValues(valueForCol);
            tids.and(equalC2); // leave only tids that are equal

        } else if (predicate.isLT()) {

            NumericalColumnPredicateGroup numGroup = (NumericalColumnPredicateGroup) predicateicateSpace.getColumn2GroupMap().get(col);
            float valueForCol = predicate.getPredicate().getCol1().getValueAt(tid);
            IndexedPredicate lt = predicate;
            RoaringBitmap ltC2;

            if (!numGroup.isBinnedLT())
                ltC2 = lt.getLtIndex().getTidsLTValues(valueForCol);
            else
                ltC2 = lt.getLtBinnedIndex().getTidsLTValues(valueForCol);
            tids.and(ltC2); // remove tids that are equal to the value

        } else if (predicate.isLTE()) {

            NumericalColumnPredicateGroup numGroup = (NumericalColumnPredicateGroup) predicateicateSpace.getColumn2GroupMap().get(col);
            float valueForCol = predicate.getPredicate().getCol1().getValueAt(tid);
            IndexedPredicate lt = numGroup.getLt();
            RoaringBitmap ltC2;

            if (!numGroup.isBinnedLT())
                ltC2 = lt.getLtIndex().getTidsLTValues(valueForCol);
            else
                ltC2 = lt.getLtBinnedIndex().getTidsLTValues(valueForCol);
            tids.and(ltC2); // remove tids that are equal to the value

            IndexedPredicate eq = numGroup.getEq();
            RoaringBitmap equalC2 = eq.getEqualityIndex().geTidsOfEqualValues(valueForCol);
            tids.and(equalC2); // leave only tids that are equal

        } else if (predicate.isGT()) {

            NumericalColumnPredicateGroup numGroup = (NumericalColumnPredicateGroup) predicateicateSpace.getColumn2GroupMap().get(col);
            float valueForCol = predicate.getPredicate().getCol1().getValueAt(tid);
            IndexedPredicate lt = numGroup.getLt();
            RoaringBitmap ltC2;

            if (!numGroup.isBinnedLT())
                ltC2 = lt.getLtIndex().getTidsLTValues(valueForCol);
            else
                ltC2 = lt.getLtBinnedIndex().getTidsLTValues(valueForCol);
            tids.andNot(ltC2); // remove tids that are equal to the value

            IndexedPredicate eq = numGroup.getEq();
            RoaringBitmap equalC2 = eq.getEqualityIndex().geTidsOfEqualValues(valueForCol);
            tids.andNot(equalC2); // leave only tids that are equal

        } else if (predicate.isGTE()) {
            NumericalColumnPredicateGroup numGroup = (NumericalColumnPredicateGroup) predicateicateSpace.getColumn2GroupMap().get(col);
            float valueForCol = predicate.getPredicate().getCol1().getValueAt(tid);
            IndexedPredicate lt = numGroup.getLt();
            RoaringBitmap ltC2;

            if (!numGroup.isBinnedLT())
                ltC2 = lt.getLtIndex().getTidsLTValues(valueForCol);
            else
                ltC2 = lt.getLtBinnedIndex().getTidsLTValues(valueForCol);
            tids.andNot(ltC2); // remove tids that are equal to the value
        }


    }

    // equality is based on the predicateicateID
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Refiner that = (Refiner) o;
        return predicate.getPredicateId() == that.predicate.getPredicateId();
    }

    @Override
    public int hashCode() {
        return Objects.hash(predicate.getPredicateId());
    }
}