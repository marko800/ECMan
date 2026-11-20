import sys
import polars as pl
import pandas as pd
import re
import statistics
from fnmatch import fnmatchcase
from pathlib import Path
import code

########################################################################################################################
# global settings
########################################################################################################################

# set threshold to check column entries for
THRESHOLD = 20

# set max nr of cols and rows to be displayed
NR_ROWS = 10
NR_COLUMNS = 30

# number parsing (used for all file types)
NUMERIC_SHARE = 1.0  # share of non-missing cells that must be numeric for a string column to be cast
DECIMAL_COMMA = False  # True: "1,5" = 1.5 and "1.234,5" = 1234.5

pl.Config.set_tbl_rows(NR_ROWS)
pl.Config.set_tbl_cols(NR_COLUMNS)


########################################################################################################################
# helper functions
########################################################################################################################

def lowercase_unique(columns: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    '''
    Lower-cases column names. If this produces duplicates, add suffix (n, n_2, n_3, ...).
    :param columns: list of column names.
    :return: new column names and a list of (original, new) for every renamed column.
    '''
    used, new_cols, renamed = set(), [], []
    for col in columns:
        low = col.lower()
        candidate, k = low, 1
        while candidate in used:
            k += 1
            candidate = f"{low}_{k}"
        used.add(candidate)
        new_cols.append(candidate)
        if candidate != low:
            renamed.append((col, candidate))
    return new_cols, renamed


def expand_columns(tokens: list[str], columns: list[str], wildcards: list[str] = None) -> tuple[list, list]:
    '''
    If a token contains *, find elements of columns that match the token with * replaced by arbitrary string.
    :param tokens: list of tokens (containing *).
    :param columns: list of columns that are potentially matched.
    :param wildcards: consider only certain columns (e.g. numeric).
    :return: list of matched columns and unmatched tokens.
    '''
    pool = columns if wildcards is None else wildcards
    selected, unmatched = [], []
    for token in tokens:
        # find all columns that match token if * is replaced by arbitrary string
        if "*" in token:
            matches = [col for col in pool if fnmatchcase(col, token)]
        else:
            matches = [token] if token in columns else []
        # if success, add matches to list of matched columns, else to list of unmatched tokens
        if matches:
            selected += [match for match in matches if match not in selected]
        else:
            unmatched.append(token)
    return selected, unmatched


########################################################################################################################
# number parsing (shared by csv, xlsx, tex)
########################################################################################################################

MISSING = {"", "-", "--", "---", "–", "—", ".", "..", "n.a.", "n/a", "na", "nan", "null", "none", "None"}


def to_number(s: str, decimal_comma: bool = DECIMAL_COMMA):
    '''int / float if s looks numeric, None if s means "missing", else s unchanged.'''
    t = s.strip()
    if t.lower() in MISSING:
        return None
    t = re.sub(r"\*+$", "", t).strip()  # 1.23***
    t = re.sub(r"^[(\[]\s*(.*?)\s*[)\]]$", r"\1", t)  # (0.12) / [0.12]
    t = t.replace("\u2212", "-").rstrip("%").strip()
    if re.fullmatch(r"[-+]?\d{1,3}( \d{3})+([.,]\d+)?", t):  # 1 234
        t = t.replace(" ", "")
    if decimal_comma:
        if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+(,\d+)?", t):
            t = t.replace(".", "")
        t = t.replace(",", ".")
    elif re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", t):
        t = t.replace(",", "")
    if re.fullmatch(r"[-+]?\d+", t):
        return int(t)
    if re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", t):
        return float(t)
    return s


def build_column(name: str, values: list[str], min_share: float, decimal_comma: bool) -> pl.Series:
    '''Numeric Series if enough cells parse as numbers, otherwise a string Series.'''
    parsed = [to_number(v, decimal_comma) for v in values]
    present = [(v, p) for v, p in zip(values, parsed) if p is not None]
    bad = [v for v, p in present if isinstance(p, str)]
    n_num = len(present) - len(bad)
    if n_num and n_num / len(present) >= min_share:
        if bad:
            print(f"WARNING: column {name!r}: {len(bad)} non-numeric cell(s) set to null, e.g. {bad[:5]}")
        data = [None if isinstance(p, str) else p for p in parsed]
        is_int = all(isinstance(p, int) for p in data if p is not None)
        return pl.Series(name, data, dtype=pl.Int64 if is_int else pl.Float64)
    if n_num and bad:
        print(f"NOTE: column {name!r} kept as text, {len(bad)} non-numeric cell(s), e.g. {bad[:5]}")
    return pl.Series(name, [v if v.strip() else None for v in values], dtype=pl.String)


def numeric_columns(df: pl.DataFrame, min_share: float = NUMERIC_SHARE,
                    decimal_comma: bool = DECIMAL_COMMA) -> pl.DataFrame:
    '''
    Cast string columns to Int64/Float64 if enough cells are numeric (see to_number / build_column).
    Columns that are already numeric are left untouched.
    :param df: pl.DataFrame, typically with all columns read as strings.
    :return: pl.DataFrame.
    '''
    cols = []
    for name in df.columns:
        s = df[name]
        if s.dtype == pl.String:
            cols.append(build_column(name, ["" if v is None else v for v in s.to_list()], min_share, decimal_comma))
        else:
            cols.append(s)
    return pl.DataFrame(cols)


########################################################################################################################
# excel parsing
########################################################################################################################

def read_excel(path: str) -> list[tuple[str, pl.DataFrame]]:
    '''
    Parse all sheets of an excel file into pl.DataFrames.
    :param path: Path to .xlsx file.
    :return: list of (sheet name, pl.DataFrame).
    '''
    # read & generate raw dfs, all cells as strings (numbers are cast afterwards by numeric_columns)
    sheets = pd.read_excel(
        path,
        sheet_name=None,
        header=None,
        engine="openpyxl",
        dtype=str
    )
    dfs = []
    for sheet_name, raw in sheets.items():
        df = None
        # drop fully empty rows
        raw = raw.dropna(how="all").reset_index(drop=True)
        if raw.empty:
            print(f"Sheet {sheet_name!r} is empty, skipped.")
            continue
        # get rid of title rows at the top (= leading rows with at most one value)
        n_title = 0
        while n_title < len(raw) and raw.iloc[n_title].notna().sum() <= 1:
            n_title += 1
        raw = raw[n_title:].reset_index(drop=True)
        # find column names (all cells are strings now, so this is the first row after the title)
        for i, row in raw.iterrows():
            if all(isinstance(x, str) for x in row.dropna()):
                col_names = raw.iloc[i].fillna("NA").astype(str).str.strip()
                df = raw[i + 1:]
                df.columns = lowercase_unique(list(col_names))[0]
                break
        if df is None:
            raise ValueError(f"no table found in sheet {sheet_name!r}")
        dfs.append((sheet_name, numeric_columns(pl.from_pandas(df))))
    return dfs


########################################################################################################################
# tex parsing
########################################################################################################################

TABLE_ENV = re.compile(r"\\begin\{(tabular\*?|tabularx|longtable)\}(.*?)\\end\{\1\}", re.DOTALL)
RULES = re.compile(
    r"\\(?:hline|toprule|midrule|bottomrule|endhead|endfirsthead|endfoot|endlastfoot)\b"
    r"|\\addlinespace(?:\[[^\]]*\])?"
    r"|\\(?:cline|cmidrule|hspace\*?|vspace\*?)(?:\([^)]*\))?\{[^}]*\}"
)
ROW_END = re.compile(r"\\\\\*?(?:\s*\[[^\]]*\])?")  # \\  \\*  \\[2pt]
CELL_SEP = re.compile(r"(?<!\\)&")  # & but not \&
WRAPPERS = re.compile(
    r"\\(?:textbf|textit|textrm|textsf|texttt|textsc|emph|text|mathrm|mathbf|mathit|mbox"
    r"|num|sym|ensuremath|underline)\s*\{([^{}]*)\}"
)
MULTIROW = re.compile(r"\\multirow\s*\{[^}]*\}\s*\{[^}]*\}\s*\{([^{}]*)\}")


def _skip_group(s: str, i: int) -> int:
    '''Index right after the balanced {...} group that starts at s[i].'''
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


def _strip_env_head(env: str, body: str) -> str:
    '''Remove [pos], {width} and the {column spec} right after \\begin{...}.'''
    body = body.lstrip()
    if env in ("tabular*", "tabularx") and body.startswith("{"):
        body = body[_skip_group(body, 0):].lstrip()  # {width}
    if body.startswith("["):
        body = body[body.find("]") + 1:].lstrip()  # [t]
    if body.startswith("{"):
        body = body[_skip_group(body, 0):]  # {lcc}, also handles @{}, p{2cm}
    return body


def find_tex_tables(text: str) -> list[str]:
    '''
    Extract the bodies of all tabular / tabular* / tabularx / longtable environments.
    Commented-out tables are ignored.
    '''
    text = re.sub(r"(?<!\\)%.*", "", text)
    tables = [_strip_env_head(m.group(1), m.group(2)) for m in TABLE_ENV.finditer(text)]
    print(f"Found {len(tables)} tex-table(s):\n")
    return tables


def _expand_multicolumn(cell: str) -> list[str]:
    '''\\multicolumn{n}{spec}{text} -> [text, "", ...] (n cells, so the row keeps its length).'''
    m = re.match(r"\s*\\multicolumn\s*\{(\d+)\}", cell)
    if not m:
        return [cell]
    rest = cell[m.end():].lstrip()
    if not rest.startswith("{"):
        return [cell]
    rest = rest[_skip_group(rest, 0):].lstrip()  # drop {spec}
    if not rest.startswith("{"):
        return [cell]
    end = _skip_group(rest, 0)
    return [rest[1:end - 1]] + [""] * (int(m.group(1)) - 1)


def clean_cell(cell: str) -> str:
    '''Strip LaTeX markup from one cell, keep the visible text.'''
    s = cell
    prev = None
    while prev != s:  # nested \textbf{\num{..}}
        prev = s
        s = MULTIROW.sub(r"\1", s)
        s = WRAPPERS.sub(r"\1", s)
    s = re.sub(r"\\textsuperscript\{[^{}]*\}|\^\{?\*+\}?", "", s)  # significance stars as superscript
    s = re.sub(r"(?<!\\)\$", "", s)  # math mode delimiters
    s = (s.replace(r"\$", "$").replace(r"\%", "%").replace(r"\&", "&")
         .replace(r"\_", "_").replace(r"\#", "#").replace("{,}", ","))
    s = re.sub(r"\\textminus\s*", "-", s)
    s = re.sub(r"\\textless\s*", "<", s)
    s = re.sub(r"\\textgreater\s*", ">", s)
    s = re.sub(r"\\textasteriskcentered\s*", "*", s)
    s = re.sub(r"\\[,;:! ]|\\q?quad\b|~", " ", s)  # spacing macros
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)  # any other leftover command
    s = s.replace("{", "").replace("}", "")
    return " ".join(s.split())


def make_col_name(header: str, i: int) -> str:
    name = re.sub(r"[\W_]+", "_", header.lower()).strip("_")
    return name or f"col{i}"


def tex_tables_to_df(tables: list[str]) -> list[pl.DataFrame]:
    '''Turn tabular bodies into pl.DataFrames (each tex row becomes one df column).'''
    dfs = []
    for t, table in enumerate(tables, 1):
        body = RULES.sub("", table)
        rows = []
        for raw_row in ROW_END.split(body):
            if not raw_row.strip():
                continue
            cells = []
            for cell in CELL_SEP.split(raw_row):
                cells += _expand_multicolumn(cell)
            cells = [clean_cell(c) for c in cells]
            if any(cells):
                rows.append(cells)
        if not rows:
            print(f"Table {t}: no rows found, skipped.")
            continue

        n_cols = statistics.mode(len(r) for r in rows)  # most common length, not the max
        kept = [r for r in rows if len(r) == n_cols]
        if len(kept) < len(rows):
            print(f"Table {t}: dropped {len(rows) - len(kept)} row(s) with deviating number of cells.")
        if n_cols < 2:
            print(f"Table {t}: no data rows left, skipped.")
            continue

        # each tex row becomes one df column, first cell = name, rest = values (all strings, cast afterwards)
        names = [make_col_name(r[0], i) for i, r in enumerate(kept)]
        names, _ = lowercase_unique(names)
        dfs.append(numeric_columns(pl.DataFrame([
            pl.Series(name, [v if v.strip() else None for v in r[1:]], dtype=pl.String)
            for name, r in zip(names, kept)
        ])))
    return dfs


########################################################################################################################
# main functions
########################################################################################################################

# standard extension types to be considered
extensions_dict = {
    ".csv": None,
    ".dta": None,
    ".json": pl.read_json,
    ".parquet": pl.read_parquet,
    ".xlsx": None
}


def check_data(df, extension=None, threshold=THRESHOLD) -> str:
    '''
    Check df for values < threshold in sensitive columns.
    :param df: pl.DataFrame.
    :param extension: Type of file where df was extracted from.
    :param threshold: Integer, minimal value allowed in sensitive columns.
    :return: Check result as string (GOOD, BAD or CHECK).
    '''
    # set all col names to lower case
    new_cols, renamed = lowercase_unique(df.columns)
    df.columns = new_cols
    for old, new in renamed:
        print(f"\nDuplicate column names after lower-casing: {old!r} renamed to {new!r}!")

    if extension == ".json":
        print("Displaying the json-file:")
        df.glimpse()
    else:
        if len(df) <= NR_ROWS:
            print("Displaying ALL data:\n")
            if len(df.columns) > NR_COLUMNS:
                df.glimpse()
            else:
                print(df)
        else:
            print("Displaying data SAMPLE:\n")
            if len(df.columns) > NR_COLUMNS:
                df.sample(3).glimpse()
            else:
                print(df.sample(3))
        print("")
    print("Column names: ", df.columns)
    cols = input(
        '''\nColumn(s) to check? Enter names separated by ',' / use '*' as placeholder for arbitrary strings / type
 'fine' if no check needed / type 'inspect' to inspect manually / press ENTER to skip: ''')
    try:
        cols = cols.lower().split(",")
        if cols == ["fine"]:
            result = "GOOD"
        elif cols == [""]:
            print("\n     ->SKIPPED: MANUAL CHECK REQUIRED!")
            result = "CHECK"
        elif cols == ["inspect"]:
            print(
                '''
 
                * Entering console - access data as 'df', type exit() or press Ctrl+Z to leave console.
                  File will be automatically passed onto check-manually list! *
 
            '''
            )
            local_vars = {"df": df, "pl": pl, "pd": pd, "exit": sys.exit, "quit": sys.exit}
            result = "CHECK"
            try:
                code.interact(local=local_vars)
            except SystemExit:
                pass
        else:
            tokens = [col.replace("'", "").strip() for col in cols]
            numeric = [col for col, type in df.schema.items() if type.is_numeric()]
            cols, unmatched = expand_columns(tokens, df.columns, numeric)
            for col in unmatched:
                print(f"\nNO (NUMERIC) COLUMN MATCHES {col!r}, MANUAL CHECK REQUIRED!")
            if cols:
                print("\nChecking columns", cols, ":")
            found_problem = False
            col_error = bool(unmatched)
            for col in cols:
                try:
                    df_filtered = df.filter((pl.col(col) < threshold) & (pl.col(col) > 0))
                except Exception as e:
                    print(f"\nCHECK OF COLUMN {col} FAILED: {e}")
                    col_error = True
                    continue
                if not df_filtered.is_empty():
                    print(
                        f"\n     -> RESULT: FOUND {len(df_filtered)} ROWS (OF {len(df)}) WITH {col} < {threshold}: \n",
                        df_filtered.sample(min(3, len(df_filtered))))
                    found_problem = True
                else:
                    print("\n     -> RESULT: ALL GOOD.")
            result = "BAD" if found_problem else ("CHECK" if col_error else "GOOD")
    except:
        print("\n     ->WRONG INPUT: MANUAL CHECK REQUIRED!")
        result = "CHECK"
    print("\n*******************************************************************************************************\n")
    return result


def open_tabular(item: str) -> str:
    '''
    Opens a file, displays column names and asks which are to be checked for case nrs < threshold.
    :param item: file path.
    :return: Evaluation of (all) table(s) in item.
    '''
    print(f"Reviewing {item}:")
    print("")
    extension = item.suffix
    read = False

    if extension == ".dta":
        try:
            df = pd.read_stata(item, convert_categoricals=False)
            df = pl.from_pandas(df)
            read = True
        except Exception as e:
            print(f"Couldn't read the dta-file ... ({e})\n")

    elif extension == ".csv":
        # find correct separator
        separators = [",", ";", "\t", "|", "="]
        success = False
        for sep in separators:
            try:
                # read all columns as strings (infer_schema_length=0), so that no cell is silently turned into null
                df = pl.read_csv(item, separator=sep, infer_schema_length=0)
                if df.width > 1:
                    success = True
                    read = True
                    break
            except Exception:
                continue
        if success:
            df = numeric_columns(df)
        else:
            print("Couldn't read the csv-file ...")

    elif extension == ".xlsx":
        try:
            sheets = read_excel(item)
        except Exception as e:
            print(f"Couldn't read the xlsx-file ... ({e})\n")
            return "CHECK"
        results = []
        for sheet_name, sheet_df in sheets:
            print(f"Sheet: {sheet_name}")
            results.append(check_data(sheet_df, extension, threshold=THRESHOLD))
        if "BAD" in results: return "BAD"
        if "CHECK" in results or not results: return "CHECK"
        return "GOOD"

    elif extension == ".json":
        try:
            df = extensions_dict[extension](item)
            read = True
        except Exception as e:
            print(f"Couldn't read the json-file ... ({e})")

    elif extension == ".parquet":
        try:
            df = extensions_dict[extension](item)
            read = True
        except Exception as e:
            print(f"Couldn't read the parquet-file ... ({e})\n")

    if read:
        return check_data(df, extension=extension, threshold=THRESHOLD)
    else:
        return "CHECK"


def check_tex(item: str) -> str:
    '''
    Extract all tables from tex file, run check_data on each table.
    :param item: file path.
    :return: Check result as string.
    '''
    print(f"Reviewing {item}:")
    print("")
    need_check, found_problem = False, False
    try:
        results = []
        with open(item, encoding="utf-8", errors="replace") as r:
            text = r.read()
            tables = find_tex_tables(text)
            dfs = tex_tables_to_df(tables)
            if not dfs:
                results = ["CHECK"]
            else:
                for i, df in enumerate(dfs):
                    results.append(check_data(df, extension=None, threshold=THRESHOLD))
                    if len(dfs) > 1 and i < len(dfs) - 1:
                        print("\nNext table:\n")
        for res in results:
            if res == "BAD":
                found_problem = True
            elif res == "CHECK":
                need_check = True

    except:
        print("Couldn't read the tex-file ...")
        need_check = True

    if found_problem:
        return "BAD"
    else:
        if need_check:
            return "CHECK"
        else:
            return "GOOD"


########################################################################################################################
# RUN
########################################################################################################################

if __name__ == "__main__":
    # print welcome screen
    print(
        '''
            ***********************************************
            *                                             *
            *           Welcome to ECMan,                 *
            *      the Export Checking Manager!           *
            *                                             *
            ***********************************************    
        '''
    )
    # determine top_folder location
    top_folder = input("Enter path to export folder: ")
    print("")
    # generate list of all files in top_folder
    file_list = [item for item in Path(top_folder).rglob("*") if item.is_file()]

    # initialise dict of check results with "not checked" as default value
    results_dict = {item: "NOGO" for item in file_list}
    # check each file
    for item in file_list:
        if item.suffix in extensions_dict.keys():
            results_dict[item] = open_tabular(item)
        elif item.suffix == ".tex":
            # run custom function to extract all tables from tex, inspect as in check_tabular ...
            results_dict[item] = check_tex(item)

    # print results ordered in categories
    print("\nFinished!")
    print("\nThe following files passed the check:\n ")
    for file in [key for key, val in results_dict.items() if val == "GOOD"]:
        print(file)
    print("\n******")
    print("\nThe following files were marked as problematic:\n ")
    for file in [key for key, val in results_dict.items() if val == "BAD"]:
        print(file)
    print("\n******")
    print("\nThe following files need a manual check:\n ")
    for file in [key for key, val in results_dict.items() if val == "CHECK"]:
        print(file)
    print("\n******")
    print("\nThe following files were not checked:\n ")
    for file in [key for key, val in results_dict.items() if val == "NOGO"]:
        print(file)
    print("\n******")
