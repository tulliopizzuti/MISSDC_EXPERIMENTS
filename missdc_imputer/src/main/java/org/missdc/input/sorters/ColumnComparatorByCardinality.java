package org.missdc.input.sorters;

import org.missdc.input.columns.Column;

import java.util.Comparator;


public class ColumnComparatorByCardinality implements Comparator<Column> {

	@Override
	public int compare(Column g1, Column g2) {
		return Long.compare(g1.getCardinality(),
				g2.getCardinality());
	}

}