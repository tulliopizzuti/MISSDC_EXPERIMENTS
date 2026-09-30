# missdc_imputer

A minimal, standalone Maven project for running the **MissDCv2 imputer** from the command line.

This repository contains only the code required by the DC-based MissDCv2 imputation pipeline. Experiment drivers, tests, alternative MissDC implementations, and ML fallback/model code are intentionally excluded.

## Requirements

- JDK 20 or newer
- Maven 3.x

Check your installation with:

```bash
java -version
mvn -version
```

## Build

From the project root, run:

```bash
mvn clean package
```

Maven creates an executable fat JAR containing MissDC and its dependencies:

```text
target/missdc_imputer.jar
```

## Run MissDC

Run the JAR with the CSV file to impute:

```bash
java -jar target/missdc_imputer.jar path/to/dirty.csv
```

For example:

```bash
java -jar target/missdc_imputer.jar data/adult.csv
```

### Output

MissDC writes the imputed dataset automatically **next to the input file**.

For an input file named:

```text
data/adult.csv
```

the output is:

```text
data/adult_imputed.csv
```

The input file is not modified. Observed values are copied to the output, inferred values replace missing cells, and cells for which MissDC finds no candidate remain empty.

## Input CSV format

The first row must contain the column name followed by its type.

Supported types are:

- `INT`
- `FLOAT`
- `STR` (or `STRING`)

Example:

```csv
age INT,workclass STR,hours FLOAT,income STR
39,State-gov,40,<=50K
50,Self-emp,,>50K
,Private,45,<=50K
```

Missing values must be represented by **empty CSV fields**, as in the example above.

Column names should not contain spaces because MissDC reads each header as `<column-name> <type>`.

## Default configuration

Running only with the input filename uses the defaults defined by `MissDCv2`:

```text
DC discovery:                 exact
Minimum evidence multiplicity: 8
Maximum DC size:                7
Voting strategy:                tuple
ML fallback:                    disabled / not included
```

Equivalent explicit command:

```bash
java -jar target/missdc_imputer.jar data/adult.csv \
  --mode dc \
  --voting tuple \
  --min-evidence-multiplicity 8 \
  --dc-size-max 7
```

## Command-line options

```text
--mode <dc|approx>
--voting <current|tuple|dc>
--min-evidence-multiplicity <n>
--dc-size-max <k|max>
```

### Examples

Exact discovery with the reproducibility defaults:

```bash
java -jar target/missdc_imputer.jar data/adult.csv \
  --mode dc \
  --voting tuple \
  --min-evidence-multiplicity 8 \
  --dc-size-max 7
```

Exact discovery without a DC-size bound:

```bash
java -jar target/missdc_imputer.jar data/adult.csv \
  --mode dc \
  --dc-size-max max
```

Approximate discovery requires an unbounded DC size:

```bash
java -jar target/missdc_imputer.jar data/adult.csv \
  --mode approx \
  --dc-size-max max
```

## Open in IntelliJ IDEA

1. Open the `missdc_imputer` directory in IntelliJ IDEA.
2. Import/open it as a Maven project if prompted.
3. Use JDK 20 (or newer) as the Project SDK.
4. Let IntelliJ/Maven download the dependencies from `pom.xml`.
5. Run `mvn clean package`, or run `org.missdc.cli.MissDCImputerRunner` directly from IntelliJ.

## Project entry points

The core imputer is:

```text
org.missdc.imputation.MissDCv2
```

The executable console entry point is:

```text
org.missdc.cli.MissDCImputerRunner
```

The fat JAR manifest is configured to use `MissDCImputerRunner`, so no classpath or main-class argument is needed when using `java -jar`.
