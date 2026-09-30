package org.missdc.input.columns;


import org.missdc.utils.misc.UnsortedObject2FloatMapProvider;
import it.unimi.dsi.fastutil.floats.Float2ObjectMap;
import it.unimi.dsi.fastutil.floats.FloatList;

import java.util.*;

public class CategoricalColumn extends Column {


    private UnsortedObject2FloatMapProvider<String> floatDictionary; // to save memory

    private Map<String, Integer> stringValues;

    public CategoricalColumn(String tableName, String name, int index, ColumnPrimitiveType colType) {

        super(tableName, name, index, colType);

        floatDictionary = new UnsortedObject2FloatMapProvider<String>();

    }

    public CategoricalColumn(String tableName, String name, int index, ColumnPrimitiveType colType,
                             boolean useSimilarity) {

        super(tableName, name, index, colType);

        floatDictionary = new UnsortedObject2FloatMapProvider<String>();

    }

    public CategoricalColumn(String tableName, String columnName, int colID, ColumnPrimitiveType colType,
                             UnsortedObject2FloatMapProvider<String> copiedfloatDictionary, FloatList copiedValuesList) {
        super(tableName, columnName, colID, colType, copiedValuesList);

        this.floatDictionary = copiedfloatDictionary;


    }


    public void addValues(ArrayList<String> values) {

        for (String value : values) {

            //if (value.isEmpty() || value.isBlank()) {
            if (value.isEmpty()) {
                value = DEFAULT_NULL_STRING;
            }

            float floatValue = floatDictionary.createOrGetFloat(value);
            valuesList.add(floatValue);
        }
    }

    public UnsortedObject2FloatMapProvider<String> getFloatDictionary() {
        return floatDictionary;
    }

    public void setFloatDictionary(UnsortedObject2FloatMapProvider<String> floatDictionary) {
        this.floatDictionary = floatDictionary;
    }

    @Override
    public int size() {
        return valuesList.size();
    }

    public Float2ObjectMap<String> buildReverseDictionaryMap() {

        return floatDictionary.getReversedMap();
    }

    public boolean containsSameValue(Column otherColumn, double percentage) {

        if (!otherColumn.ColumnType.equals(this.ColumnType))
            return false;

        Map<String, Integer> thisCounts = getStringValues();

        Map<String, Integer> otherColumnCounts = ((CategoricalColumn) otherColumn).getStringValues();


        int totalCount = 0;
        int sharedCount = 0;
        for (String s : thisCounts.keySet()) {
            int thisCount = thisCounts.get(s);
            Integer otherCount = otherColumnCounts.get(s);
            if (otherCount == null)
                otherCount = 0;
            sharedCount += Math.min(thisCount, otherCount);
            totalCount += Math.max(thisCount, otherCount);
        }


        return ((double) sharedCount) / ((double) totalCount) > percentage;

    }

    @Override
    public List<Integer> getTIDsMissing() {
        if (tidsMissing == null) {

            tidsMissing = new ArrayList<>();

            float floatValueForNull = getFloatDictionary().getFloatForString(DEFAULT_NULL_STRING);

            if (floatValueForNull == 0)
                return tidsMissing; // no null has been seen in that columns

            for (int tid = 0; tid < valuesList.size(); tid++) {
                if (valuesList.getFloat(tid) == floatValueForNull) {

                    tidsMissing.add(tid);
                }
            }
        }
        return tidsMissing;
    }

    @Override
    public List<Integer> rebuildTIDsMissing() {


        tidsMissing = new ArrayList<>();

        float floatValueForNull = getFloatDictionary().getFloatForString(DEFAULT_NULL_STRING);

        if (floatValueForNull == 0)
            return tidsMissing; // no null has been seen in that columns

        for (int tid = 0; tid < valuesList.size(); tid++) {
            if (valuesList.getFloat(tid) == floatValueForNull) {

                tidsMissing.add(tid);
            }
        }

        return tidsMissing;
    }

    public boolean containsMissing() {
        List<Integer> tids = getTIDsMissing();
        return tids != null && !tids.isEmpty();
    }


    @Override
    public Set<Float> getDomain() {
        if (domain != null)
            return domain;

        domain = new HashSet<>();

        float floatValueForNull = getFloatDictionary().getFloatForString(DEFAULT_NULL_STRING);

        for (float v : valuesList) {
            if (v != floatValueForNull)
                domain.add(v);
        }


        return domain;
    }

    public Map<String, Integer> getStringValues() {

        if (stringValues == null) {
            stringValues = new HashMap<>();

            for (float f : valuesList) {
                String value = floatDictionary.getObject(f);
                int count = stringValues.containsKey(value) ? stringValues.get(value) : 0;
                stringValues.put(value, count + 1);
            }
        }

        return stringValues;
    }

//	@Override
//	public String getStrValueAt(int tid) {
//		
//		return floatDictionary.getObject(index);
//	}

    public void clearValuesCountMap() {
        stringValues = null;

    }



    public String toString() {
        // sb.append(TableName + "." + ColumnName);


        return ColumnName;
    }


}