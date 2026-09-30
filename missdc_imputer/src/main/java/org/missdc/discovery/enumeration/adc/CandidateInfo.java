package org.missdc.discovery.enumeration.adc;

import java.util.BitSet;

public class CandidateInfo implements Comparable<CandidateInfo> {

    BitSet candidate;
    BitSet eviIDs;
    long count;

    public CandidateInfo(BitSet candidate, BitSet remainingEvi, long vioCount) {
        this.candidate = candidate;
        this.eviIDs = remainingEvi;
        this.count = vioCount;
    }

    @Override
    public int compareTo(CandidateInfo o) {
        // compare based on count
        return Long.compare(this.count, o.count);
    }
}