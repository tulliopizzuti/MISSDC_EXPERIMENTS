package org.missdc.utils.bitset;

import java.util.BitSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public class BitUtils {

    public static BitSet buildBitSet(int pid) {
        BitSet bs = new BitSet();
        bs.set(pid);
        return bs;
    }

    public static BitSet buildBitSetMinusBit(BitSet bs, int minusBit) {
        BitSet newBs = (BitSet) bs.clone();
        newBs.clear(minusBit);
        return newBs;
    }

    public static BitSet getAndBitSet(BitSet pBits, BitSet eviBits) {

        BitSet andBits;

        // TODO check if cloning smaller is even worth it
        if (pBits.size() < eviBits.size()) {
            andBits = (BitSet) pBits.clone();
            andBits.and(eviBits);
        } else {
            andBits = (BitSet) eviBits.clone();
            andBits.and(pBits);
        }

        return andBits;
    }

    public static BitSet getOrBitSet(BitSet bs1, BitSet bs2) {

        BitSet orBits = (BitSet) bs1.clone();
        orBits.or(bs2);


        return orBits;
    }

    public static BitSet getAndNotBitSet(BitSet bs1, BitSet bs2) {

        BitSet andNotBits = (BitSet) bs1.clone();

        andNotBits.andNot(bs2);

        return andNotBits;
    }

    public static BitSet getIntersection(BitSet bs1, BitSet bs2) {
        BitSet inter = (BitSet) bs1.clone();
        inter.and(bs2);
        return inter;
    }

    public static BitSet copyAndAddBit(BitSet oldBitset, int bit2add) {
        BitSet newBitset = (BitSet) oldBitset.clone();
        newBitset.set(bit2add);

        return newBitset;
    }

    public static boolean isContainedIn(BitSet predPath, BitSet evi) {

        BitSet bs = (BitSet) predPath.clone();

        bs.and(evi);

        return bs.cardinality() == predPath.cardinality();
    }

    // for LongBitSet
    public static IBitSet buildIBitSet(int pid) {

        IBitSet bs = LongBitSet.FACTORY.create();
        bs.set(pid);

        return bs;
    }

    public static boolean isSubSetOf(BitSet preds, BitSet evi) {

        BitSet bsAnd = (BitSet) preds.clone();
        bsAnd.and(evi);
        return bsAnd.cardinality() == preds.cardinality();
    }

    public static void removeFromList(List<Integer> subPathPredicateList, BitSet coveredPredicates) {

        //iterate the bits in coveredPredicate
        for (Integer pid = coveredPredicates.nextSetBit(0); pid >= 0; pid = coveredPredicates.nextSetBit(pid + 1)) {
            subPathPredicateList.remove(pid);
        }
    }


    public static Set<BitSet> getSubSetWithIntersection(Set<BitSet> bitsets, BitSet bitset) {

        Set<BitSet> hasIntersection = new LinkedHashSet<>();

        for (BitSet bs : bitsets) {

            if (bs.intersects(bitset)) {
                hasIntersection.add(bs);
            }

        }

        return hasIntersection;

    }

    public static Set<BitSet> getSubSetWithId(Set<BitSet> bitsets, int predicateId) {

        Set<BitSet> containsPredicate = new LinkedHashSet<>();

        for (BitSet bs : bitsets) {

            if (bs.get(predicateId)) {
                containsPredicate.add(bs);
            }

        }

        return containsPredicate;
    }

    public static Set<BitSet> getIntersectingSubset(Set<BitSet> dcs, BitSet predicates4Col) {

        Set<BitSet> containsPredicate = new LinkedHashSet<>();

        for (BitSet dc : dcs) {

            if (dc.intersects(predicates4Col)) {
                containsPredicate.add(dc);
            }

        }

        return containsPredicate;
    }
}