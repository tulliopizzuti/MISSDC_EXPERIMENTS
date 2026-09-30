package org.missdc.discovery;

import org.missdc.dc.predicates.space.PredicateSpace;
import org.missdc.dc.predicates.space.builder.PredicateSpaceBuilder;
import org.missdc.discovery.enumeration.DCEnumeration;
import org.missdc.discovery.enumeration.HybridEvidenceInversion;
import org.missdc.discovery.enumeration.adc.ADC;
import org.missdc.discovery.evidence.EvidenceSet;
import org.missdc.discovery.evidence.context.IEvidenceSetBuilder;
import org.missdc.discovery.evidence.context.RowECP;
import org.missdc.input.Table;
import org.missdc.input.reader.CSVInput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.BitSet;
import java.util.Set;

public class DCDiscoverer {

    private static final Logger log = LoggerFactory.getLogger(DCDiscoverer.class);
    private static final double approxThreshold = 0.0001d;
    private final boolean useApproximateDCs;
    private EvidenceSet evidenceSet;
    private final CSVInput input;

    public DCDiscoverer(String dataset, boolean useApproximateDCs) throws Exception {
        this.input = new CSVInput(dataset);
        this.useApproximateDCs = useApproximateDCs;
    }

    public DCDiscoverer(String dataset) throws Exception {
        this(dataset, false);
    }

    public EvidenceSet getEvidenceSet() {
        return evidenceSet;
    }


    public Set<BitSet> run() throws Exception {
        return run(1, HybridEvidenceInversion.UNBOUNDED);
    }


    public Set<BitSet> run(long minEvidenceMultiplicity) throws Exception {
        return run(minEvidenceMultiplicity, HybridEvidenceInversion.UNBOUNDED);
    }


    public Set<BitSet> run(long minEvidenceMultiplicity, int maxDCSize) throws Exception {

        if (minEvidenceMultiplicity < 1) {
            throw new IllegalArgumentException("minEvidenceMultiplicity must be >= 1");
        }
        if (maxDCSize != HybridEvidenceInversion.UNBOUNDED && maxDCSize < 1) {
            throw new IllegalArgumentException("maxDCSize must be >= 1 or UNBOUNDED (-1)");
        }
        if (useApproximateDCs && maxDCSize != HybridEvidenceInversion.UNBOUNDED) {
            throw new IllegalArgumentException(
                    "dc-size-max is currently supported only for exact DC discovery");
        }

        log.info("DC Discovery on {} ...", input.getDataFileName());

        Table table = input.getTable();
        PredicateSpace predicateSpace = new PredicateSpaceBuilder().build(table);

        log.info("Building evidence set...");

        IEvidenceSetBuilder evidenceSetBuilder;

        if (minEvidenceMultiplicity > 1) {

            log.info("Filtering out evidence patterns with min multiplicity = {}", minEvidenceMultiplicity);
            evidenceSetBuilder = new RowECP(table, predicateSpace, minEvidenceMultiplicity);

        } else {
            evidenceSetBuilder = new RowECP(table, predicateSpace);
        }

        long evidenceBuildStart = System.nanoTime();

        this.evidenceSet = evidenceSetBuilder.build();

        long evidenceBuildMs =
                (System.nanoTime() - evidenceBuildStart) / 1_000_000L;

        log.info("Size of evi set = {}", this.evidenceSet.size());
        log.info("Evidence building time: {} ms", evidenceBuildMs);



        if (useApproximateDCs) {

            log.info("Searching for approximate DCs...");
            DCEnumeration dcEnumeration = new ADC(predicateSpace, evidenceSet, approxThreshold);
            Set<BitSet> dcs = dcEnumeration.searchDCs();
            log.info("{} DCs found", dcs.size());

            return dcs;

        } else {

            if (maxDCSize == HybridEvidenceInversion.UNBOUNDED) {
                log.info("Searching for DCs...");
            } else {
                log.info("Searching for DCs with size <= {}...", maxDCSize);
            }

            long enumerationStart = System.nanoTime();

            DCEnumeration dcEnumeration =
                    new HybridEvidenceInversion(predicateSpace, evidenceSet, maxDCSize);

            Set<BitSet> dcs = dcEnumeration.searchDCs();

            long enumerationMs =
                    (System.nanoTime() - enumerationStart) / 1_000_000L;

            log.info("{} DCs found", dcs.size());
            log.info("Evidence enumeration time: {} ms", enumerationMs);

            return dcs;
        }
    }

}