package org.missdc.imputation.voting;

import java.util.Locale;

/**
 * Voting semantics compared by the MissDC voting experiment.
 *
 * CURRENT - every (DC, donor tuple) occurrence contributes one vote.
 * TUPLE   - every distinct donor tuple contributes at most one vote.
 * DC      - every DC contributes one vote for the local majority of its donors.
 */
public enum VotingStrategy {
    CURRENT,
    TUPLE,
    DC;

    public static VotingStrategy parse(String value) {
        return switch (value.toLowerCase(Locale.ROOT).replace('-', '_')) {
            case "current", "pair", "pair_vote", "dc_tuple" -> CURRENT;
            case "tuple", "tuple_vote", "unique_tuple" -> TUPLE;
            case "dc", "dc_vote", "constraint" -> DC;
            default -> throw new IllegalArgumentException(
                    "Unsupported voting strategy '" + value
                            + "'. Use current, tuple, or dc.");
        };
    }

    public String fileTag() {
        return name().toLowerCase(Locale.ROOT);
    }
}
