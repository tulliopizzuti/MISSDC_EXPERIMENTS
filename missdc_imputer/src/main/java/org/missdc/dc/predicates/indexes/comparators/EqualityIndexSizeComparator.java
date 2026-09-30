package org.missdc.dc.predicates.indexes.comparators;


import org.missdc.dc.predicates.groups.ColumnPredicateGroup;

import java.util.Comparator;

public class EqualityIndexSizeComparator implements Comparator<ColumnPredicateGroup> {

	@Override
	public int compare(ColumnPredicateGroup g1, ColumnPredicateGroup g2) {
		return Integer.compare(g1.getEq().getEqualityIndex().getIndex().size(),
				g2.getEq().getEqualityIndex().getIndex().size());
	}

}