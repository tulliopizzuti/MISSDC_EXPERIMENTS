package org.missdc.input.columns;


import org.missdc.utils.misc.MurmurHash3;
import it.unimi.dsi.fastutil.floats.FloatArrayList;
import it.unimi.dsi.fastutil.floats.FloatList;
import net.agkn.hll.HLL;
import org.roaringbitmap.RoaringBitmap;

import java.util.*;

public abstract class Column {

    public static final int DEFAULT_INITIAL_LIST_CAPACITY = 10000;

    public static final String DEFAULT_NULL_STRING = "<NULL>";
//public static final String DEFAULT_NULL_STRING = "";

    public static final Float DEFAULT_NULL_NUMBER = Float.MIN_VALUE;
//    public static final Float DEFAULT_NULL_NUMBER = -100000.0f;
    public final String TableName;
    public final String ColumnName;
    public final int ColumnIndex;

    public final ColumnPrimitiveType ColumnType;
    protected FloatList valuesList;
    protected HLL hll;
    protected long cardinality = 0;
    protected List<Integer> tidsMissing = null;

    protected RoaringBitmap bitmapOfMissingTids = null;
    protected Set<Float> domain;
    Float mode = null;
    List<Float> valueOrderByFreq = null;

    public Column(String tableName, String colName, int colIndex, ColumnPrimitiveType colType) {
        this.TableName = tableName;
        this.ColumnName = colName;
        this.ColumnIndex = colIndex;
        this.ColumnType = colType;
        // this.nextTid = 0;

        valuesList = new FloatArrayList(DEFAULT_INITIAL_LIST_CAPACITY);
    }


    public Column(String tableName, String colName, int colIndex, ColumnPrimitiveType colType, FloatList valuesList) {
        this.TableName = tableName;
        this.ColumnName = colName;
        this.ColumnIndex = colIndex;
        this.ColumnType = colType;
        // this.nextTid = 0;

        this.valuesList = valuesList;
    }

    public abstract Set<Float> getDomain();

    public abstract void addValues(ArrayList<String> values);

    public abstract boolean containsSameValue(Column c1, double percentage);

    public abstract List<Integer> getTIDsMissing();

//	public abstract String getStrValueAt(int tid);

    public abstract List<Integer> rebuildTIDsMissing();

    public float getValueAt(int i) {

        return valuesList.getFloat(i);

    }

    public int size() {

        return valuesList.size();
    }

    public long getCardinality() {

        if (hll == null) {

            hll = new HLL(13/* log2m */, 5/* registerWidth */);

            for (float value : valuesList) {

                hll.addRaw(MurmurHash3.fmix64(Float.floatToRawIntBits(value)));

            }

            cardinality = hll.cardinality();

        }

        return cardinality;
    }

    public FloatList getValuesList() {
        return valuesList;
    }

    float mostFrequent = -1;
    public float mostFrequent(){
        if(mostFrequent == -1){
            Map<Float, Integer> countMap = new HashMap<>();
            for (float value : valuesList) {
                countMap.put(value, countMap.getOrDefault(value, 0) + 1);
            }
            int max = 0;
            for (Map.Entry<Float, Integer> entry : countMap.entrySet()) {
                if(entry.getValue() > max){
                    max = entry.getValue();
                    mostFrequent = entry.getKey();
                }
            }
        }
        return mostFrequent;
    }

    public void setValuesList(FloatList orderedValues) {
        this.valuesList = orderedValues;
    }

    public void setValuesListBuildNewHLL(FloatList newValues) {
        this.valuesList = newValues;
        hll = new HLL(13/* log2m */, 5/* registerWidth */);

        for (float value : valuesList) {

            hll.addRaw(MurmurHash3.fmix64(Float.floatToRawIntBits(value)));

        }

        cardinality = hll.cardinality();

    }

    public String toString() {
        // sb.append(TableName + "." + ColumnName);
        //		if (valuesList != null && !valuesList.isEmpty())
//			sb.append(": " + valuesList);

        return ColumnName
//		if (valuesList != null && !valuesList.isEmpty())
//			sb.append(": " + valuesList);
                ;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ColumnIndex;
        result = prime * result + ((ColumnName == null) ? 0 : ColumnName.hashCode());
        result = prime * result + ((ColumnType == null) ? 0 : ColumnType.hashCode());
        result = prime * result + ((TableName == null) ? 0 : TableName.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Column other = (Column) obj;
        if (ColumnIndex != other.ColumnIndex)
            return false;
        if (ColumnName == null) {
            if (other.ColumnName != null)
                return false;
        } else if (!ColumnName.equals(other.ColumnName))
            return false;
        if (ColumnType != other.ColumnType)
            return false;
        if (TableName == null) {
            return other.TableName == null;
        } else return TableName.equals(other.TableName);
    }

    public String getSQLTypedColString() {

        return "\"" + ColumnName + " " + ColumnType + "\" " + mapToSQLType(ColumnType);
    }

    public String getCSVTypedColString() {

        return "\"" + ColumnName + " " + ColumnType + "\"";
    }

    public String getSQLServerTypedColString() {

        return ColumnName + " " + mapToSQLType(ColumnType);
    }

    private String mapToSQLType(ColumnPrimitiveType columnType) {

        switch (columnType) {
            case INT:
                return "integer";
            case FLOAT:
                return "decimal(12,4)";
            case STR:
                return "varchar(500)";

        }

        return null;

    }

    public RoaringBitmap getBitmapOfMissingTids() {
        if (bitmapOfMissingTids != null)
            return bitmapOfMissingTids;

        bitmapOfMissingTids = new RoaringBitmap();
        for (Integer tid : getTIDsMissing()) {
            bitmapOfMissingTids.add(tid);
        }

        return bitmapOfMissingTids;
    }

    public abstract void clearValuesCountMap();

    public List<Float> getValueOrderByFreq() {

        if (valueOrderByFreq == null) {

            FloatList floatList = valuesList;

            Map<Float, Integer> countMap = new HashMap<>();
            for (float value : floatList) {
                countMap.put(value, countMap.getOrDefault(value, 0) + 1);
            }
            //order coutMap by value
            valueOrderByFreq = new ArrayList<>(countMap.keySet());
            Collections.sort(valueOrderByFreq, new Comparator<Float>() {
                @Override
                public int compare(Float o1, Float o2) {
                    return countMap.get(o2).compareTo(countMap.get(o1));
                }
            });
        }

        return valueOrderByFreq;


    }

    public List<Float> getMode(int n) {
        List<Float> orderedValuesByFre = getValueOrderByFreq();

        if (n <= orderedValuesByFre.size()) {
            return orderedValuesByFre.subList(0, n);
        } else {
            return orderedValuesByFre;
        }

    }

    public enum ColumnPrimitiveType {
        INT, FLOAT, STR
    }

}