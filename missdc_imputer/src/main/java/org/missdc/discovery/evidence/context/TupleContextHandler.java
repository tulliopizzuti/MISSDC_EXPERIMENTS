package org.missdc.discovery.evidence.context;


import org.missdc.dc.predicates.IndexedPredicate;
import org.missdc.dc.predicates.groups.ColumnPredicateGroup;
import org.missdc.dc.predicates.groups.NumericalColumnPredicateGroup;
import org.missdc.discovery.evidence.EvidenceSet;
import org.roaringbitmap.RoaringBitmap;

import java.util.*;

public class TupleContextHandler {

    long startTime, endTime, timeElapsed;
    private final int tid;
    private Set<RhsContext> contextSet;
    private Set<RhsContext> contexts2add;
    private Set<RhsContext> contexts2remove;

    public TupleContextHandler(int tuple, RoaringBitmap tableIds, BitSet baseEvidence) {
        super();
        this.tid = tuple;
        this.contextSet = new HashSet<>();

        RhsContext initialContext = createContext(tableIds, baseEvidence);
        this.contextSet.add(initialContext);

        this.contexts2add = new HashSet<>();
        this.contexts2remove = new HashSet<>();

    }

    private TupleContextHandler(int tuple, RhsContext context) {
        this.tid = tuple;
        this.contextSet = new HashSet<>();
        contextSet.add(context);
        this.contexts2add = new HashSet<>();
        this.contexts2remove = new HashSet<>();

    }

    private RhsContext createContext(RoaringBitmap tableIds, BitSet baseEvidence) {

        RoaringBitmap c2 = tableIds.clone();
        c2.remove(tid);

        BitSet initialEvidence = (BitSet) baseEvidence.clone();

        RhsContext context = new RhsContext(c2, initialEvidence);

        return context;
    }

//	public static long timeIntersec=0;

    public void updateCategoricalContext(ColumnPredicateGroup catGroup) {

        IndexedPredicate eq = catGroup.getEq();

        float tupleValue = eq.getPredicate().getCol1().getValueAt(tid);

        RoaringBitmap equalC2 = eq.getEqualityIndex().geTidsOfEqualValues(tupleValue);

        if (equalC2.getCardinality() == 1) { // singleton tuples, everyone is different
            return;
        }

        contexts2add.clear();
        contexts2remove.clear();

        Iterator<RhsContext> contextIt = contextSet.iterator();

        while (contextIt.hasNext()) {

            RhsContext context = contextIt.next();

            RoaringBitmap sharedEqual = RoaringBitmap.and(equalC2, context.c2);

            if (sharedEqual.getCardinality() > 0) {

                if (context.c2.equals(sharedEqual)) {// all c2 has an equal value, we can remove the !=
                    contexts2remove.add(context);
                } else {
                    context.c2.andNot(sharedEqual);
                }

                BitSet newEvidence = (BitSet) context.evidence.clone();

                newEvidence.xor(eq.getEqCategoricalMask());// put the equal into the evidence

                RhsContext newContext = new RhsContext(sharedEqual, newEvidence);

                contexts2add.add(newContext);

            }

        }

        contextSet.addAll(contexts2add);
        contextSet.removeAll(contexts2remove);

    }

    public void updateNumericalContext(NumericalColumnPredicateGroup numGroup) {

        float tupleValue = numGroup.getColumn().getValueAt(tid);

        IndexedPredicate eq = numGroup.getEq();
        IndexedPredicate lt = numGroup.getLt();

        RoaringBitmap equalC2 = eq.getEqualityIndex().geTidsOfEqualValues(tupleValue);

        RoaringBitmap ltC2;

        if (!numGroup.isBinnedLT())
            ltC2 = lt.getLtIndex().getTidsLTValues(tupleValue);
        else
            ltC2 = lt.getLtBinnedIndex().getTidsLTValues(tupleValue);

        contexts2add.clear();
        contexts2remove.clear();

        Iterator<RhsContext> contextIt = contextSet.iterator();

//		System.out.println("tuple: " + tid + " Tuple value: " + tupleValue + " with tids " + ltC2);

        while (contextIt.hasNext()) {

            RhsContext context = contextIt.next();

            boolean skipRange = false;

            RoaringBitmap sharedEqual = RoaringBitmap.and(equalC2, context.c2);

            if (sharedEqual.getCardinality() > 0) {

                if (context.c2.equals(sharedEqual)) {// all c2 has an equal value, we can remove the !=
                    skipRange = true; // no need to test other cases
                    contexts2remove.add(context);
                } else {
                    context.c2.andNot(sharedEqual);
                }

                BitSet newEvidence = (BitSet) context.evidence.clone();

                newEvidence.xor(eq.getEqRangeMaskLt());// put the equal (and gte/lte) into the evidence

                RhsContext newContext = new RhsContext(sharedEqual, newEvidence);

                contexts2add.add(newContext);

            }

            if (!skipRange) {

                RoaringBitmap sharedLt = RoaringBitmap.and(ltC2, context.c2);

                if (sharedLt.getCardinality() > 0) {

                    BitSet newEvidence = (BitSet) context.evidence.clone();

                    newEvidence.xor(lt.getRangeMask());// put the inverse range predicates into the evidence

                    RhsContext newContext = new RhsContext(sharedLt, newEvidence);

                    contexts2add.add(newContext);

                    context.c2.andNot(sharedLt);

                    if (context.c2.getCardinality() == 0) {//

                        contexts2remove.add(context);
                    }

                }

            }

        }

        contextSet.addAll(contexts2add);
        contextSet.removeAll(contexts2remove);

    }

    public void collectEvidence(EvidenceSet evidenceSet,
                                List<NumericalColumnPredicateGroup> numericalColPredicateGroups) {

        Iterator<RhsContext> contextIt = contextSet.iterator();

        if (numericalColPredicateGroups.isEmpty()) {

            while (contextIt.hasNext()) {
                RhsContext context = contextIt.next();

                evidenceSet.addEvidenceWithMultiplicity(context.evidence, context.c2.getLongCardinality() * 2);

            }

        } else {

            while (contextIt.hasNext()) {
                RhsContext context = contextIt.next();

                long eviMultiplicity = context.c2.getLongCardinality();

                if (eviMultiplicity == 0) {
                    continue;
                }

                evidenceSet.addEvidenceWithMultiplicity(context.evidence, eviMultiplicity);

                BitSet invertedEvidence = (BitSet) context.evidence.clone();

                for (NumericalColumnPredicateGroup ngroup : numericalColPredicateGroups) {

                    if (invertedEvidence.get(ngroup.getUneq().getPredicateId())) {

                        // note that using xor(ngroup.getGt().getRangeRangeMask() would result in the
                        // same result (check how the mask is built in case there is a question)
                        invertedEvidence.xor(ngroup.getLt().getRangeMask()); // we simple invert using the xor mask

                    }
                }

                evidenceSet.addEvidenceWithMultiplicity(invertedEvidence, eviMultiplicity);

            }

        }

    }

    public void collectEvidenceFixMissing(EvidenceSet evidenceSet,
                                          List<NumericalColumnPredicateGroup> numericalColPredicateGroups, Map<ColumnPredicateGroup, RoaringBitmap> group2TidsWithNulls) {


        // fix for missing values

        contexts2add.clear();
        contexts2remove.clear();

        for (Map.Entry<ColumnPredicateGroup, RoaringBitmap> entry : group2TidsWithNulls.entrySet()) {


            ColumnPredicateGroup group = entry.getKey();
            BitSet predicates = group.getPredicateIDs();

            RoaringBitmap tidsWithNulls = entry.getValue();

            Iterator<RhsContext> contextIt = contextSet.iterator();

            while (contextIt.hasNext()) {

                RhsContext context = contextIt.next();

                RoaringBitmap tidsWithMissingInContext = RoaringBitmap.and(context.c2, tidsWithNulls);

                if (!tidsWithMissingInContext.isEmpty()) {

                    BitSet newEvidence = (BitSet) context.evidence.clone();

                    newEvidence.andNot(predicates); // remove predicates for that group

                    RhsContext newContext = new RhsContext(tidsWithMissingInContext, newEvidence);

                    contexts2add.add(newContext);

                    context.c2.andNot(tidsWithMissingInContext);

                    if (context.c2.isEmpty()) {//

                        contexts2remove.add(context);
                    }
                }


            }

            contextSet.addAll(contexts2add);
            contextSet.removeAll(contexts2remove);

            contexts2add.clear();
            contexts2remove.clear();


        }


        Iterator<RhsContext> contextIt = contextSet.iterator();

        if (numericalColPredicateGroups.isEmpty()) {

            while (contextIt.hasNext()) {
                RhsContext context = contextIt.next();

                evidenceSet.addEvidenceWithMultiplicity(context.evidence, context.c2.getLongCardinality() * 2);

            }

        } else {

            while (contextIt.hasNext()) {
                RhsContext context = contextIt.next();

                long eviMultiplicity = context.c2.getLongCardinality();

                if (eviMultiplicity == 0) {
                    continue;
                }

                evidenceSet.addEvidenceWithMultiplicity(context.evidence, eviMultiplicity);

                BitSet invertedEvidence = (BitSet) context.evidence.clone();

                for (NumericalColumnPredicateGroup ngroup : numericalColPredicateGroups) {

                    if (invertedEvidence.get(ngroup.getUneq().getPredicateId())) {

                        // note that using xor(ngroup.getGt().getRangeRangeMask() would result in the
                        // same result (check how the mask is built in case there is a question)
                        invertedEvidence.xor(ngroup.getLt().getRangeMask()); // we simple invert using the xor mask

                    }
                }

                evidenceSet.addEvidenceWithMultiplicity(invertedEvidence, eviMultiplicity);

            }

        }


    }

    public Set<RhsContext> getContextSet() {
        return contextSet;
    }

    public List<TupleContextHandler> splitContexts() {

        List<TupleContextHandler> contexts = new ArrayList<TupleContextHandler>();

        Iterator<RhsContext> contextIt = contextSet.iterator();

        while (contextIt.hasNext()) {

            RhsContext context = contextIt.next();
            contexts.add(new TupleContextHandler(this.tid, context));
        }

        return contexts;
    }

    public void clear() {
        contextSet = null;
        this.contexts2add = null;
        this.contexts2remove = null;

    }


}