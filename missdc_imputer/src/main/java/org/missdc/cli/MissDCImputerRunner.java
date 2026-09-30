package org.missdc.cli;

import org.missdc.discovery.enumeration.HybridEvidenceInversion;
import org.missdc.imputation.MissDCv2;
import org.missdc.imputation.voting.VotingStrategy;

/** Console entry point for the reproducibility build of MissDCv2. */
public final class MissDCImputerRunner {

    private static final String USAGE =
            "Usage: java -jar missdc_imputer.jar <dirty-file.csv> "
                    + "[--mode <dc|approx>] "
                    + "[--voting <current|tuple|dc>] "
                    + "[--min-evidence-multiplicity <n>] "
                    + "[--dc-size-max <k|max>]";

    private MissDCImputerRunner() {}

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            throw new IllegalArgumentException(USAGE);
        }

        String dirtyFile = args[0];
        boolean approximate = false;
        VotingStrategy voting = MissDCv2.DEFAULT_VOTING_STRATEGY;
        long minEvidenceMultiplicity = MissDCv2.DEFAULT_MIN_EVIDENCE_MULTIPLICITY;
        int maxDCSize = MissDCv2.DEFAULT_MAX_DC_SIZE;

        for (int i = 1; i < args.length; i++) {
            String flag = args[i];
            if (++i >= args.length) {
                throw new IllegalArgumentException("Missing value after " + flag + ". " + USAGE);
            }
            String value = args[i];

            switch (flag) {
                case "--mode" -> approximate = parseMode(value);
                case "--voting" -> voting = VotingStrategy.parse(value);
                case "--min-evidence-multiplicity" ->
                        minEvidenceMultiplicity = parseMultiplicity(value);
                case "--dc-size-max" -> maxDCSize = parseMaxDCSize(value);
                default -> throw new IllegalArgumentException(
                        "Unknown option '" + flag + "'. " + USAGE);
            }
        }

        if (approximate && maxDCSize != HybridEvidenceInversion.UNBOUNDED) {
            throw new IllegalArgumentException(
                    "Approximate DC discovery does not support bounded DC size. "
                            + "Use --dc-size-max max.");
        }

        new MissDCv2(
                dirtyFile,
                approximate,
                minEvidenceMultiplicity,
                maxDCSize,
                voting).run();
    }

    private static boolean parseMode(String value) {
        return switch (value.toLowerCase()) {
            case "dc" -> false;
            case "approx" -> true;
            default -> throw new IllegalArgumentException(
                    "Unsupported mode '" + value + "'. Use dc or approx.");
        };
    }

    private static long parseMultiplicity(String value) {
        try {
            long parsed = Long.parseLong(value);
            if (parsed < 1) {
                throw new IllegalArgumentException(
                        "Minimum evidence multiplicity must be >= 1.");
            }
            return parsed;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid evidence multiplicity '" + value + "'.", e);
        }
    }

    private static int parseMaxDCSize(String value) {
        if ("max".equalsIgnoreCase(value) || "unbounded".equalsIgnoreCase(value)) {
            return HybridEvidenceInversion.UNBOUNDED;
        }

        try {
            int parsed = Integer.parseInt(value);
            if (parsed < 1) {
                throw new IllegalArgumentException("Maximum DC size must be >= 1.");
            }
            return parsed;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid maximum DC size '" + value + "'.", e);
        }
    }
}
