package org.missdc.input.sorters;

import org.missdc.input.columns.Column;

import java.util.Comparator;



public class RowComparatorOneColumn implements Comparator<Integer> {

	private final Column column2order;

	public RowComparatorOneColumn(Column column2order) {
		super();
		this.column2order = column2order;
	}

	@Override
	public int compare(Integer i1, Integer i2) {
		return Float.compare(column2order.getValueAt(i1), column2order.getValueAt(i2));
	}

}