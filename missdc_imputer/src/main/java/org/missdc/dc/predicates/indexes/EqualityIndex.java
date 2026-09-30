package org.missdc.dc.predicates.indexes;

import org.missdc.dc.predicates.IndexedPredicate;
import org.missdc.dc.predicates.Predicate;
import org.missdc.input.columns.Column;
import it.unimi.dsi.fastutil.floats.Float2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.floats.FloatList;
import org.roaringbitmap.RoaringBitmap;


public class EqualityIndex {

	private final Predicate predicate;

	private final Float2ObjectOpenHashMap<RoaringBitmap> index;

	public EqualityIndex(IndexedPredicate indexedPredicate) {

		this.predicate = indexedPredicate.getPredicate();

		Column col = predicate.getCol1();
		FloatList valuesList = col.getValuesList();

		// System.out.println(valuesList);

		index = new Float2ObjectOpenHashMap<RoaringBitmap>((int) col.getCardinality());

		for (int tid = 0; tid < valuesList.size(); tid++) {

			float value = valuesList.getFloat(tid);

			RoaringBitmap mapEntry = index.computeIfAbsent(value, bitmap -> new RoaringBitmap());

			mapEntry.add(tid);
		}

	}

	public Float2ObjectOpenHashMap<RoaringBitmap> getIndex() {
		return index;
	}

	public RoaringBitmap geTidsOfEqualValues(float value) {
		return index.get(value);
	}

	@Override
	public String toString() {
		return "Index [index=" + index + "]";
	}

	public Predicate getPredicate() {

		return predicate;
	}

	public void removeTids(RoaringBitmap handledTids) {

		for (RoaringBitmap tids : index.values()) {
			tids.andNot(handledTids);
		}
	}

	public void removeValue(float nullFloatValue) {

		index.remove(nullFloatValue);
	}
}