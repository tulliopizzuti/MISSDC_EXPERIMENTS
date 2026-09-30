package org.missdc.discovery.enumeration.adc;


import org.missdc.utils.bitset.BitUtils;
import it.unimi.dsi.fastutil.objects.Object2LongOpenHashMap;

import java.util.*;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

public class SubpathValidator implements Callable<Set<BitSet>> {

    //parent
    final ADC adc;
    // child
    protected List<BitSet> sortedReshapedEvidenceList;
    protected Map<Integer, BitSet> pid2BitEviIDsMap; // for each predicate p, a BitSet with the eviIDs of evidence containing p
    List<Integer> sortedPredicateList;
    List<Integer> candidatePositions;
    CandidateInfo parentCandidate;
    Object2LongOpenHashMap<BitSet> reshapedEvidence;

    Set<BitSet> subPathDCs;


    public SubpathValidator(ADC adc, List<Integer> sortedPredicateList, List<Integer> key, CandidateInfo value) {
        this.adc = adc;
        this.sortedPredicateList = sortedPredicateList;
        this.candidatePositions = key;
        this.parentCandidate = value;
    }

    public Set<BitSet> searchDCs() {

        subPathDCs = new HashSet<>();

        int nextCandPos = candidatePositions.get(candidatePositions.size() - 1);

        if (nextCandPos == sortedPredicateList.size()) {
            return subPathDCs;
        }

        // prepare the list of predicate for that path
        List<Integer> subPathPredicateList = new ArrayList<>(sortedPredicateList);
        subPathPredicateList = subPathPredicateList.subList(nextCandPos, subPathPredicateList.size());

        BitSet pidsCovered = removeCoveredPredicates(sortedPredicateList, candidatePositions, subPathPredicateList);

        reshapeEvidence(parentCandidate, pidsCovered);

        buildPid2BitEviIDsMap(subPathPredicateList);

        Map<Integer, CandidateInfo> predPos2CandidateMap = checkNextLevelCandidates(subPathPredicateList, parentCandidate); // discover DCs of size 3 and returns the remaining predicates in order of remaining vio (descending)

        if (predPos2CandidateMap.isEmpty()) {
            return subPathDCs;
        }

        // sort pred2Candi by vio count, the key integers are positions of subPathPredicateList
        List<Map.Entry<Integer, CandidateInfo>> sortedPredPos2Candi = predPos2CandidateMap.entrySet().stream()
                .sorted((e1, e2) -> Long.compare(e2.getValue().count, e1.getValue().count))
                .collect(Collectors.toList());

        for (Map.Entry<Integer, CandidateInfo> entry : sortedPredPos2Candi) {
            int pidPos = entry.getKey();
            int pid = subPathPredicateList.get(pidPos);

            List<Integer> newSubPathPredicateList = new ArrayList<>(subPathPredicateList);
            newSubPathPredicateList = newSubPathPredicateList.subList(pidPos + 1, newSubPathPredicateList.size());

            newSubPathPredicateList.removeAll(adc.pred2PredGroupMap.get(pid)); // remove implied predicates

            CandidateInfo subpathCandidateInfo = entry.getValue();
            recursiveSearch(newSubPathPredicateList, subpathCandidateInfo);
        }
        return subPathDCs;

    }

    private void recursiveSearch(List<Integer> subPathPredicateList, CandidateInfo candidate) {

        if (subPathPredicateList.isEmpty())
            return;

        if (candidate.count <= adc.maxVioNumber) {
            subPathDCs.add(candidate.candidate);
            return;
        }

//        if (candidate.candidate.cardinality() >= 7) {
//            return;
//        }

        BitSet preds2Cover = new BitSet();
        subPathPredicateList.forEach(preds2Cover::set);

        pidforcandidate:
        for (int pidPos = 0; pidPos < subPathPredicateList.size(); pidPos++) {

            int pid = subPathPredicateList.get(pidPos);

            BitSet newRemainingEvi = (BitSet) candidate.eviIDs.clone();
            newRemainingEvi.and(pid2BitEviIDsMap.get(pid));

            long newVioCount = 0;
            long impossibleEviVioCount = 0;

            for (int eviID = newRemainingEvi.nextSetBit(0); eviID >= 0; eviID = newRemainingEvi.nextSetBit(eviID + 1)) {
                BitSet evi = sortedReshapedEvidenceList.get(eviID);
                long count = reshapedEvidence.getLong(evi);

                if (BitUtils.isSubSetOf(preds2Cover, evi)) {
                    impossibleEviVioCount += count;
                    // no further candidate will be able to reduce  the violations to become a valid ADC
                    if (impossibleEviVioCount > adc.maxVioNumber) {
                        continue pidforcandidate;
                    }
                }

                newVioCount += count;
            }

            BitSet newCandidate = (BitSet) candidate.candidate.clone();
            newCandidate.set(pid);


            if (newVioCount <= adc.maxVioNumber) {
                subPathDCs.add(newCandidate);
            } else {

                CandidateInfo newCandidateInfo = new CandidateInfo(newCandidate, newRemainingEvi, newVioCount);

                List<Integer> newSubPathPredicateList = new ArrayList<>(subPathPredicateList);
                newSubPathPredicateList = newSubPathPredicateList.subList(pidPos + 1, newSubPathPredicateList.size());
                newSubPathPredicateList.removeAll(adc.pred2PredGroupMap.get(pid)); // remove implied predicates

                recursiveSearch(newSubPathPredicateList, newCandidateInfo);
            }
        }

    }


    private Map<Integer, CandidateInfo> checkNextLevelCandidates(List<Integer> subPathPredicateList, CandidateInfo candidateInfo) {

        Map<Integer, CandidateInfo> pred2CandidateMap = new HashMap<>();

        BitSet preds2Cover = new BitSet();
        subPathPredicateList.forEach(preds2Cover::set);

        pidforcandidate:
        for (int predPos = 0; predPos < subPathPredicateList.size(); predPos++) {
            int pid = subPathPredicateList.get(predPos);

            BitSet pidEvis = pid2BitEviIDsMap.get(pid);
            BitSet remainingEvi = BitUtils.getAndBitSet(candidateInfo.eviIDs, pidEvis);

            long vioCount = 0;
            long impossibleEviVioCount = 0;

            for (int eviID = remainingEvi.nextSetBit(0); eviID >= 0; eviID = remainingEvi.nextSetBit(eviID + 1)) {

                BitSet evi = sortedReshapedEvidenceList.get(eviID);
                long count = reshapedEvidence.getLong(evi);

                if (BitUtils.isSubSetOf(preds2Cover, evi)) {
                    impossibleEviVioCount += count;
                    // no further candidate will be able to reduce  the violations to become a valid ADC
                    if (impossibleEviVioCount > adc.maxVioNumber) {
                        continue pidforcandidate;
                    }
                }
                vioCount += count;
            }

            BitSet newCandidate = (BitSet) candidateInfo.candidate.clone();
            newCandidate.set(pid);

            if (vioCount <= adc.maxVioNumber) {
                subPathDCs.add(newCandidate);
            } else {
                pred2CandidateMap.put(predPos, new CandidateInfo(newCandidate, remainingEvi, vioCount));
            }

        }

        return pred2CandidateMap;

    }

    //TODO make reshape optional depending on how many evi we have
    private void reshapeEvidence(CandidateInfo candidateInfo, BitSet pidsCovered) {


        reshapedEvidence = new Object2LongOpenHashMap<>();

        for (int eviID = candidateInfo.eviIDs.nextSetBit(0); eviID >= 0; eviID = candidateInfo.eviIDs.nextSetBit(eviID + 1)) {

            BitSet originalEvi = adc.sortedEvidenceList.get(eviID);
            long count = adc.eviset.getEvidence2CountMap().getLong(originalEvi);

            BitSet reshaped = (BitSet) originalEvi.clone();
            reshaped.andNot(pidsCovered);

            reshapedEvidence.addTo(reshaped, count);

        }

        sortedReshapedEvidenceList = new ArrayList<>(reshapedEvidence.keySet());
        sortedReshapedEvidenceList.sort((e1, e2) -> {
            long count1 = reshapedEvidence.getLong(e1);
            long count2 = reshapedEvidence.getLong(e2);
            return Long.compare(count2, count1);
        });

        // the subpathcandidate contain all initial (reshaped) eviIDs
        candidateInfo.eviIDs = new BitSet();
        candidateInfo.eviIDs.set(0, sortedReshapedEvidenceList.size());


    }

    private void buildPid2BitEviIDsMap(List<Integer> subPathPredicateList) {

        pid2BitEviIDsMap = new HashMap<>();

        for (int pid : subPathPredicateList) {
            pid2BitEviIDsMap.put(pid, new BitSet());
        }

        for (int eviID = 0; eviID < sortedReshapedEvidenceList.size(); eviID++) {
            BitSet evi = sortedReshapedEvidenceList.get(eviID);
            for (int pID = evi.nextSetBit(0); pID >= 0; pID = evi.nextSetBit(pID + 1)) {
                BitSet pindex = pid2BitEviIDsMap.get(pID);
                if (pindex != null)
                    pindex.set(eviID);
            }
        }


    }

    public BitSet removeCoveredPredicates(List<Integer> sortedPredicateList, List<Integer> candidatePositions, List<Integer> subPathPredicateList) {

        BitSet pidsCovered = new BitSet();
        for (Integer candidatePosition : candidatePositions) {
            int pid = sortedPredicateList.get(candidatePosition);
            Set<Integer> covered = adc.pred2PredGroupMap.get(pid);
            subPathPredicateList.removeAll(covered);
            covered.forEach(pidsCovered::set);
        }

        return pidsCovered;

    }

    @Override
    public Set<BitSet> call() throws Exception {
        return searchDCs();
    }
}