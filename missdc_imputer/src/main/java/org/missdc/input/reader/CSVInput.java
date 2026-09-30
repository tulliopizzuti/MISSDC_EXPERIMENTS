package org.missdc.input.reader;


import org.missdc.input.Table;
import org.missdc.input.columns.CategoricalColumn;
import org.missdc.input.columns.Column;
import org.missdc.input.columns.NumericalColumn;
import org.missdc.input.sorters.ColumnComparatorByCardinality;
import org.missdc.input.sorters.RowComparatorMultiColumn;
import org.missdc.input.sorters.RowComparatorOneColumn;
import it.unimi.dsi.fastutil.floats.FloatArrayList;
import it.unimi.dsi.fastutil.floats.FloatList;
import org.simpleflatmapper.csv.CsvParser;
import org.simpleflatmapper.util.CloseableIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Reads the header of dataset from CSV files
 */
public class CSVInput {

    protected static Logger log = LoggerFactory.getLogger(CSVInput.class);
    protected final String dataFileName;
    protected Table table;
    protected long loadingTime;

    protected int maximumNumColumns = -1;// -1 means (all) columns
    protected Set<String> requiredColNames = null; // null means (all) columns

    public CSVInput(String dataFileName, int maximumNRows) throws Exception {
        this.dataFileName = dataFileName;
        prepareInput(maximumNRows);
    }

    public CSVInput(String dataFileName) throws Exception {
        this.dataFileName = dataFileName;
        prepareInput();
    }

    public CSVInput(String dataFileName, int maximumNRows, int maximumColumns) throws Exception {
        this.dataFileName = dataFileName;
        this.maximumNumColumns = maximumColumns;
        prepareInput(maximumNRows);
    }

    public CSVInput(String dataFileName, int maximumNRows, Set<String> requiredColNames) throws Exception {
        this.dataFileName = dataFileName;
        this.requiredColNames = requiredColNames;
        prepareInput(maximumNRows);
    }

    public CSVInput(String dataFileName, Set<String> requiredColNames) throws Exception {
        this.dataFileName = dataFileName;
        this.requiredColNames = requiredColNames;
        prepareInput();
    }

    private void prepareInput() throws Exception {
        prepareInput(-1);
    }

    private void prepareInput(int maximumNRows) throws Exception {

        File tableFile = new File(dataFileName);

        log.info("Reading the input file: " + tableFile.getName());
        Map<Integer, Column> allColumns = new HashMap<>();
        Map<Integer, NumericalColumn> numericalColumns = new HashMap<>();
        Map<Integer, CategoricalColumn> categoricalColumns = new HashMap<>();

        // Prepare the Column Objects: colName, colType and colIdx
        table = getTableWithColumnHeaders(tableFile, allColumns, numericalColumns, categoricalColumns);

        long startTime = System.currentTimeMillis();

        if (maximumNRows > 0)
            log.info("Limiting the number of rows by: " + maximumNRows);
        readRows(maximumNRows);

        long endTime = System.currentTimeMillis();
        loadingTime = endTime - startTime;

    }

    protected Table getTableWithColumnHeaders(File tableFile, Map<Integer, Column> allColumns,
                                              Map<Integer, NumericalColumn> numericalColumns, Map<Integer, CategoricalColumn> categoricalColumns)
            throws IOException {

        String tableName = tableFile.getName().replace(".csv", "");

        Table baseTable = null;

        Map<String, Column> colName2Collumn = new HashMap<>();

        try (CloseableIterator<String[]> it = CsvParser.iterator(tableFile)) {

            if (it.hasNext()) {
                String[] headers = it.next();
                int nCols = headers.length;

                for (int colIndex = 0; colIndex < nCols; colIndex++) {
//					System.out.println(headers[colIndex]);

                    String[] headerParts = headers[colIndex].split(" ");


                    String colName = headerParts[0].replaceAll("\\s+", "").toUpperCase();
                    String colType = headerParts[1].replaceAll("\\s+", "").toUpperCase();

                    if (colType.toUpperCase().equals(Column.ColumnPrimitiveType.INT.toString())) {

                        NumericalColumn collumn = new NumericalColumn(tableName, colName, colIndex,
                                Column.ColumnPrimitiveType.INT);
                        numericalColumns.put(colIndex, collumn);
                        allColumns.put(colIndex, collumn);

                        colName2Collumn.put(colName.toLowerCase(), collumn);

                    } else if (colType.toUpperCase().equals(Column.ColumnPrimitiveType.FLOAT.toString())) {

                        NumericalColumn collumn = new NumericalColumn(tableName, colName, colIndex,
                                Column.ColumnPrimitiveType.FLOAT);
                        numericalColumns.put(colIndex, collumn);
                        allColumns.put(colIndex, collumn);

                        colName2Collumn.put(colName.toLowerCase(), collumn);

                    } else if (colType.toUpperCase().equals(Column.ColumnPrimitiveType.STR.toString())
                            || (colType.equalsIgnoreCase("STRING"))) {

                        CategoricalColumn collumn = new CategoricalColumn(tableName, colName, colIndex,
                                Column.ColumnPrimitiveType.STR);
                        categoricalColumns.put(colIndex, collumn);
                        allColumns.put(colIndex, collumn);

                        colName2Collumn.put(colName.toLowerCase(), collumn);

                    } else {
                        throw new IOException("I do not support the type " + colType + " yet.");
                    }

                }

                baseTable = new Table(tableFile, tableName, nCols, allColumns, numericalColumns, categoricalColumns);

                if (maximumNumColumns != -1 && maximumNumColumns < nCols) {

                    requiredColNames = new HashSet<>();

                    Random rand = new Random();

                    while (requiredColNames.size() < maximumNumColumns) {
                        int randomIndex = rand.nextInt(nCols);
                        requiredColNames.add(allColumns.get(randomIndex).ColumnName.toLowerCase());
                    }

                    Set<String> nonRequiredColNames = new HashSet<>(colName2Collumn.keySet());
                    nonRequiredColNames.removeAll(requiredColNames);

                    // column projection (removal)
                    for (String colName : nonRequiredColNames) {
                        allColumns.remove(colName2Collumn.get(colName).ColumnIndex);
                        numericalColumns.remove(colName2Collumn.get(colName).ColumnIndex);
                        categoricalColumns.remove(colName2Collumn.get(colName).ColumnIndex);
                    }

                    baseTable = new Table(tableFile, tableName, nCols, allColumns, numericalColumns,
                            categoricalColumns);

                }

            } else {
                throw new IOException("I could not parse the " + dataFileName + "data file");
            }

        }

        return baseTable;

    }

    private void readRows(int maximumNRows) throws IOException {

        if (maximumNRows < 0)
            CSVRowReader.readRows(table); // no limit of rows
        else
            CSVRowReader.readRows(table, maximumNRows);

        log.info("Number of rows: " + table.getNUM_RECORDS());
        log.info("Number of collumns: " + table.NUM_ORIGINAL_COLLUMNS);

    }

    public void sortCat() {

        ColumnComparatorByCardinality columnComparator = new ColumnComparatorByCardinality();

        List<CategoricalColumn> catCols = new ArrayList<>(table.getCategoricalColumns().values());
        Collections.sort(catCols, columnComparator);

        List<Integer> orderedRowIDs = IntStream.range(0, table.getNUM_RECORDS()).boxed().collect(Collectors.toList());

        RowComparatorOneColumn rowComparator = new RowComparatorOneColumn(catCols.get(0));
        Collections.sort(orderedRowIDs, rowComparator);

        for (Column c : table.getAllColumns().values()) {

            FloatList orderedValues = new FloatArrayList(table.getNUM_RECORDS());

            for (int tid : orderedRowIDs) {
                orderedValues.add(c.getValueAt(tid));

            }

            c.setValuesList(orderedValues);

        }

    }

    public void sort() {

        ColumnComparatorByCardinality columnComparator = new ColumnComparatorByCardinality();

        List<NumericalColumn> numCols = new ArrayList<>(table.getNumericalColumns().values());
        Collections.sort(numCols, columnComparator);

        List<Integer> orderedRowIDs = IntStream.range(0, table.getNUM_RECORDS()).boxed().collect(Collectors.toList());

        RowComparatorOneColumn rowComparator = new RowComparatorOneColumn(numCols.get(numCols.size() - 1));
        Collections.sort(orderedRowIDs, rowComparator);

        for (Column c : table.getAllColumns().values()) {

            FloatList orderedValues = new FloatArrayList(table.getNUM_RECORDS());

            for (int tid : orderedRowIDs) {
                orderedValues.add(c.getValueAt(tid));

            }
            c.setValuesList(orderedValues);
        }

    }

    public void sortMulti() {
        sortMulti(0);
    }

    public void sortMulti(int levels) {

        ColumnComparatorByCardinality columnComparator = new ColumnComparatorByCardinality();

        List<NumericalColumn> numCols = new ArrayList<>(table.getNumericalColumns().values());
        Collections.sort(numCols, columnComparator);
        Collections.reverse(numCols);

        if (numCols.isEmpty()) {
            return;
        }

        List<Integer> orderedRowIDs = IntStream.range(0, table.getNUM_RECORDS()).boxed().collect(Collectors.toList());

        if (levels > 0)
            numCols = numCols.subList(0, levels);

        RowComparatorMultiColumn rowComparator = new RowComparatorMultiColumn(numCols);
        Collections.sort(orderedRowIDs, rowComparator);

        for (Column c : table.getAllColumns().values()) {

            FloatList orderedValues = new FloatArrayList(table.getNUM_RECORDS());

            for (int tid : orderedRowIDs) {
                orderedValues.add(c.getValueAt(tid));

            }

            c.setValuesList(orderedValues);

        }

    }

    public void printCSVFile() throws IOException {
        printCSVFile(20);
    }

    public void printCSVFile(int NLines) throws IOException {

        log.info("Printing " + NLines + " lines of " + table.NAME + " file ...");

        try (CloseableIterator<String[]> it = CsvParser.iterator(table.DATASETFILE)) {

            int i = -1;
            while (it.hasNext()) {

                String[] values = it.next();

                if (i == -1)
                    System.out.printf("%-5s", "tid");
                else
                    System.out.printf("%-5s", i);

                for (String v : values) {
                    System.out.printf("%-15s", v);
                }

                System.out.println();

                i++;

                if (i == NLines)
                    break;

            }

        }

        // System.out.println();

    }

    public void saveImputed(String[][] imputedDataset) {

        String imputedFileName = dataFileName.replace(".csv", "_imputed.csv");

        try {
            // Open the file for reading
            FileReader dirtyFile = new FileReader(dataFileName);
            BufferedReader bufferedReader = new BufferedReader(dirtyFile);
            String header = bufferedReader.readLine();
            bufferedReader.close();

            FileWriter fileWriter = new FileWriter(imputedFileName);
            BufferedWriter writer = new BufferedWriter(fileWriter);

            writer.write(header);
            writer.newLine();

            for (int i = 0; i < imputedDataset.length; i++) {
                StringBuilder sb = new StringBuilder();

                for (int j = 0; j < imputedDataset[i].length -1; j++) {
                    // Write a CSS rule for each element in the matrix
                    if (imputedDataset[i][j] == null)
                        sb.append(",");
                    else
                        sb.append(imputedDataset[i][j] + ",");
                }

                int j = imputedDataset[i].length - 1;

                if (imputedDataset[i][j] != null)
                    sb.append(imputedDataset[i][j]);



                writer.write(sb.toString());
                writer.newLine();
            }


            writer.close();


        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    public void printColumnsValues() {
        table.printColumnsValues();
    }

    public Table getTable() {
        return table;
    }

    public void setTable(Table table) {
        this.table = table;
    }

    public String getDataFileName() {
        return dataFileName;
    }

    public long getLoadingTime() {
        return loadingTime;
    }


}