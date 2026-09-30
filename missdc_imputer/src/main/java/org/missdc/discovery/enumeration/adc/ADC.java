package org.missdc.discovery.enumeration.adc;

import org.missdc.dc.predicates.groups.ColumnPredicateGroup;
import org.missdc.dc.predicates.space.PredicateSpace;
import org.missdc.discovery.enumeration.DCEnumeration;
import org.missdc.discovery.evidence.EvidenceSet;
import org.missdc.utils.bitset.BitUtils;
import org.missdc.utils.bitset.LongBitSet;
import org.missdc.utils.search.NTreeSearch;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ADC implements DCEnumeration {

    protected final PredicateSpace pspace;
    protected final EvidenceSet eviset;
    protected final long maxVioNumber;
    protected boolean parallel = true;
    protected List<BitSet> sortedEvidenceList;
    protected Map<Integer, BitSet> pid2BitEviIDsMap; // for each predicate p, a BitSet with the eviIDs of evidence containing p
    protected Map<Integer, Long> pid2VioCountMap;
    protected Map<Integer, Set<Integer>> pred2PredGroupMap;
    Set<BitSet> dcs;

    public ADC(PredicateSpace predicateSpace, EvidenceSet evidenceSet, double approxLevel) {
        this.pspace = predicateSpace;
        this.eviset = evidenceSet;
        long eviCount = evidenceSet.getEviCounter();
        this.maxVioNumber = eviCount - (long) Math.ceil((1d - approxLevel) * (double) eviCount);
        initPredicateGroupsMap();
    }


    // generate documentation for this method
    private void initPredicateGroupsMap() {
        pred2PredGroupMap = new HashMap<>();
        for (ColumnPredicateGroup pGroup : pspace.getPredicateGroups()) {
            Set<Integer> pids = new HashSet<>();

            for (int i = pGroup.getPredicateIDs().nextSetBit(0); i != -1; i = pGroup.getPredicateIDs().nextSetBit(i + 1)) {
                pids.add(i);
            }

            for (int i = pGroup.getPredicateIDs().nextSetBit(0); i != -1; i = pGroup.getPredicateIDs().nextSetBit(i + 1)) {
                pred2PredGroupMap.put(i, pids);

            }
        }
    }

    public Set<BitSet> searchDCs() {

        dcs = new HashSet<>();

        buildIndexesAndMaps();

        Set<Integer> singlePredicates = discoverFirstLevelDCs(); // we  remove single predicate DCs from the candidates

        // predicates in ascending order of violations
        List<Integer> sortedPredicateList = IntStream.range(0, pspace.size()).boxed().sorted((p1, p2) -> {
            long vioCount1 = pid2VioCountMap.get(p1);
            long vioCount2 = pid2VioCountMap.get(p2);
            return Long.compare(vioCount1, vioCount2);
        }).collect(Collectors.toList());

        sortedPredicateList.removeAll(singlePredicates);


        Map<List<Integer>, CandidateInfo> candidate2VioCountMap = discoverSecondLevelDCs(sortedPredicateList);

        // sort by violation ascending
        Map<List<Integer>, CandidateInfo> sortedCandidate2VioCountMap = candidate2VioCountMap.entrySet().stream().sorted(Map.Entry.comparingByValue()).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (oldValue, newValue) -> oldValue, LinkedHashMap::new));

        if (!parallel) {

            for (Map.Entry<List<Integer>, CandidateInfo> entry : sortedCandidate2VioCountMap.entrySet()) {
                SubpathValidator adcExecutor = new SubpathValidator(this, sortedPredicateList, entry.getKey(), entry.getValue());
                Set<BitSet> subdcs = adcExecutor.searchDCs();
                dcs.addAll(subdcs);
            }
        } else {


            ExecutorService executor = Executors.newFixedThreadPool(8);


            try {


                List<Future<Set<BitSet>>> futuresDCs = new ArrayList<>();
                for (Map.Entry<List<Integer>, CandidateInfo> entry : sortedCandidate2VioCountMap.entrySet()) {
                    SubpathValidator adcExecutor = new SubpathValidator(this, sortedPredicateList, entry.getKey(), entry.getValue());
                    Future<Set<BitSet>> futureDCs = executor.submit(adcExecutor);
                    futuresDCs.add(futureDCs);
                }

                futuresDCs.forEach(future -> {
                    try {
                        dcs.addAll(future.get());
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                });

            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                executor.shutdown();
            }


        }

        checkMinimality();


        return dcs;
    }

    private List<List<Map.Entry<List<Integer>, CandidateInfo>>> split(Map<List<Integer>, CandidateInfo> sortedCandidate2VioCountMap) {

//        System.out.println(sortedCandidate2VioCountMap.size() + " candidates");

        List<List<Map.Entry<List<Integer>, CandidateInfo>>> splitedEntryList = new ArrayList<>();

        long totalVio = sortedCandidate2VioCountMap.values().stream().mapToLong(c -> c.count).sum();
        long vioLimit = totalVio / sortedCandidate2VioCountMap.size();

//        System.out.println("totalVio: " + totalVio + ", splitVio: " + vioLimit + ", size: " + sortedCandidate2VioCountMap.size());

        // split the task in equal sizes, according to the number of violations handled


        List<Map.Entry<List<Integer>, CandidateInfo>> entryList = new ArrayList<>();
        long actualVioLimit = 0;

        for (Map.Entry<List<Integer>, CandidateInfo> entry : sortedCandidate2VioCountMap.entrySet()) {

            entryList.add(entry);

            actualVioLimit += entry.getValue().count;

            if (actualVioLimit >= vioLimit) {
                splitedEntryList.add(entryList);
                entryList = new ArrayList<>();
                actualVioLimit = 0;
            }
        }
        return splitedEntryList;
    }


    private Set<Integer> discoverFirstLevelDCs() {

        Set<Integer> singlePredicateDCs = new HashSet<>();

        for (var entry : pid2VioCountMap.entrySet()) {

            if (entry.getValue() <= maxVioNumber) {
                int pid = entry.getKey();
//                System.out.println("DC: " + pspace.getPredicateById(pid) + " " + entry.getValue());
                BitSet dc = BitUtils.buildBitSet(pid);
                dcs.add(dc);
                singlePredicateDCs.add(pid);
            }
        }

        return singlePredicateDCs;
    }

    private Map<List<Integer>, CandidateInfo> discoverSecondLevelDCs(List<Integer> sortedPredicateList) {


        Map<List<Integer>, CandidateInfo> candidate2VioCountMap = new HashMap<>();

        for (int p1Listidx = 0; p1Listidx < sortedPredicateList.size(); p1Listidx++) {

            int p1 = sortedPredicateList.get(p1Listidx);

            Set<Integer> p1Group = pred2PredGroupMap.get(p1);

            BitSet evisP1 = pid2BitEviIDsMap.get(p1);

            List<Integer> candidatePos = new ArrayList<>();
            candidatePos.add(p1Listidx);

            for (int p2ListIdx = p1Listidx + 1; p2ListIdx < sortedPredicateList.size(); p2ListIdx++) {


                int p2 = sortedPredicateList.get(p2ListIdx);
                if (p1Group.contains(p2)) continue;

                BitSet evisP2 = pid2BitEviIDsMap.get(p2);

                BitSet remainingEvi = BitUtils.getAndBitSet(evisP1, evisP2);

                long vioCount = getVioCount(remainingEvi);

                BitSet dcCandidate = BitUtils.buildBitSet(p1);
                dcCandidate.set(p2);

                if (vioCount <= maxVioNumber) {
//                    System.out.println("DC: " + pspace.getPredicateById(p1) + " " + pspace.getPredicateById(p2) + " " + vioCount);

                    dcs.add(dcCandidate);

                } else {
                    candidatePos.add(p2ListIdx);
                    candidate2VioCountMap.put(candidatePos, new CandidateInfo(dcCandidate, remainingEvi, vioCount));

                    candidatePos = new ArrayList<>();
                    candidatePos.add(p1Listidx);
                }

            }

        }

        return candidate2VioCountMap;

    }


    private void buildIndexesAndMaps() {

        sortedEvidenceList = getSortedEviByCount();

        pid2BitEviIDsMap = getPid2BitEviIDsMap();

        pid2VioCountMap = getPid2VioCountMap();


    }

    private long getVioCount(BitSet eviIDs) {
        long count = 0;

        for (int eviID = eviIDs.nextSetBit(0); eviID >= 0; eviID = eviIDs.nextSetBit(eviID + 1)) {
            BitSet evi = sortedEvidenceList.get(eviID);
            count += eviset.getEvidence2CountMap().getLong(evi);
        }

        return count;
    }

    private Map<Integer, Long> getPid2VioCountMap() {

        Map<Integer, Long> pred2VioCountMap = new HashMap<>();

        for (int pid = 0; pid < pspace.size(); pid++) {
            pred2VioCountMap.put(pid, 0L);
        }


        for (BitSet evi : sortedEvidenceList) {
            long count = eviset.getEvidence2CountMap().getLong(evi);

            for (int pid = evi.nextSetBit(0); pid >= 0; pid = evi.nextSetBit(pid + 1))
                pred2VioCountMap.compute(pid, (k, v) -> v + count);
        }


        return pred2VioCountMap;
    }


    private Map<Integer, BitSet> getPid2BitEviIDsMap() {
        Map<Integer, BitSet> map = new HashMap<>();

        for (int pid = 0; pid < pspace.size(); pid++) {
            map.put(pid, new BitSet());
        }

        for (int eviID = 0; eviID < sortedEvidenceList.size(); eviID++) {
            BitSet evi = sortedEvidenceList.get(eviID);
            for (int pID = evi.nextSetBit(0); pID >= 0; pID = evi.nextSetBit(pID + 1)) {
                map.get(pID).set(eviID);
            }
        }

        return map;

    }

    private List<BitSet> getSortedEviByCount() {

        List<BitSet> sortedEvidence = new ArrayList<>(eviset.getEvidence2CountMap().keySet());

        sortedEvidence.sort((e1, e2) -> {
            long count1 = eviset.getEvidence2CountMap().getLong(e1);
            long count2 = eviset.getEvidence2CountMap().getLong(e2);
            return Long.compare(count2, count1);
        });
//        sortedEvidence.forEach(evi -> System.out.println(eviset.getEvidence2CountMap().getLong(evi)));
        return sortedEvidence;
    }


    private void checkMinimality() {

        Set<LongBitSet> covers = new HashSet<>();
        dcs.forEach(dc -> covers.add(LongBitSet.FACTORY.create(dc)));

        NTreeSearch nt = new NTreeSearch();
        for (LongBitSet key : covers) {
            nt.add(key);
        }
        Set<BitSet> nonGeneralized = new HashSet<>();

        for (LongBitSet bs : covers) {

            nt.remove(bs);

            if (!nt.containsSubset(bs)) {
                nonGeneralized.add(bs.toBitSet());
            }
            nt.add(bs);
        }
        dcs = nonGeneralized;
    }


}