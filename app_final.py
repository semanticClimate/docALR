# ============================================
# 1️⃣ conda create -n phytochem python=3.11 -y
# 2️⃣ conda activate phytochem
# 3️⃣ pip install -r requirements.txt
# 4️⃣ streamlit run app_final.py
# ============================================

import streamlit as st
import subprocess
import os
import sys
import shutil
import tempfile
from zipfile import ZipFile
from datetime import date
import pandas as pd
import streamlit.components.v1 as components
import networkx as nx
from st_cytoscape import cytoscape
import re
import requests
import time
# ============================================================
# AUTO-INSTALL REQUIRED LIBRARIES
# ============================================================
REQUIRED_LIBS = ["streamlit", "pandas", "networkx", "st_cytoscape"]

def ensure_installed(package):
    try:
        __import__(package)
    except ImportError:
        st.info(f"  Installing missing package: {package}")
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=False)

for lib in REQUIRED_LIBS:
    ensure_installed(lib)

# ============================================================
# PAGE SETTINGS
# ============================================================
st.set_page_config(page_title="Essential Oil Knowledge Graph", layout="wide")

st.markdown(
    "<h1 style='color:blue; font-style:italic; font-size:40px; margin: 10px; padding-left: 0px; text-align: center; '>Automated Literature Review</h1>",
    unsafe_allow_html=True
)

# ============================================================
# MAIN TABS
# ============================================================
st.markdown("""
    <style>
    /* Target Streamlit's tab buttons deeply */
    div[data-baseweb="tab-list"] button[data-baseweb="tab"] p {
        font-size: 22px !important;	 /* increase tab text size */
        font-weight: 700 !important;     /* make tab text bold */
        color: #1f1f1f !important;	 /* darker text color */
    }

    /* Highlight the active tab */
    div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] p {
        color: #52a447 !important;	 /* red text for active tab */
        border-bottom: 3px solid #e63946 !important;
        padding-bottom: 1px !important;
    }

    /* Center tabs and add spacing */
    div[data-baseweb="tab-list"] {
        justify-content: center !important;
        gap: 16px !important;
    }
    </style>
""", unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs([
    "  Download Papers", "  Summary Tables", "  Entity Extraction", " Network Visualization"
])

query = st.sidebar.text_input("Search query", "Enter Your Query")
limit = st.sidebar.number_input("Max papers", 1, 1000, 10)

# ============================================================
# TAB 1: DATA DOWNLOAD (Optimized with Live Progress + Offline)
# ============================================================
with tab1:
    col1, col2 = st.columns([1, 5])
    with col1:
	st.markdown(
            """
            <img src="https://raw.githubusercontent.com/petermr/pygetpapers/main/resources/pygetpapers_logo.png"
                 width="90" height="90">
            """,
            unsafe_allow_html=True
        )
    with col2:
	st.markdown("<h2 style='padding-top: 15px;'>Download Papers with pygetpapers</h2>", unsafe_allow_html=True)

    mode = st.radio("Select Mode:", ["  Online Fetch (via pygetpapers)", "  Offline Mode (use existing folder)"])

    # ----------------------------------------------------------------------
    # ONLINE MODE
    # ----------------------------------------------------------------------
    if mode == "  Online Fetch (via pygetpapers)":

        download_options = st.multiselect(
            "  Choose formats to download",
            options=["PDF", "XML", "HTML"],
            default=["PDF", "XML"]
        )

	api = st.selectbox("API", [
            "europe_pmc", "crossref", "arxiv", "biorxiv", "medrxiv", "rxivist", "openalex"
        ])

	use_start = st.checkbox("Start date filter")
        start_date = st.date_input("Start date", value=date(2000, 1, 1)) if use_start else None

        use_end = st.checkbox("End date filter")
        end_date = st.date_input("End date", value=date(2025, 12, 31)) if use_end else None

        if st.button("  Start Download"):
            temp_dir = tempfile.mkdtemp()
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            st.session_state['output_dir'] = output_dir

            cmd = [
                "pygetpapers",
                "-q", query,
                "-o", output_dir,
                "-k", str(limit),
                "--api", api,
                "--save_query"
            ]

            if "PDF" in download_options:
                cmd.append("-p")
            if "XML" in download_options:
                cmd.append("-x")
            if "HTML" in download_options:
                cmd.append("--makehtml")

            if start_date:
                cmd.extend(["--startdate", start_date.strftime("%Y-%m-%d")])
            if end_date:
                cmd.extend(["--enddate", end_date.strftime("%Y-%m-%d")])

            st.info("  Downloading papers... This may take several minutes.")
            progress = st.empty()
            st.write("  Fetching results, please wait...")

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            lines = []

            for line in iter(process.stdout.readline, ''):
                lines.append(line.strip())
                if len(lines) > 10:
                    lines = lines[-10:]
                progress.text("\n".join(lines))

            process.wait()

            if process.returncode != 0:
                st.error("  Download failed. Check the error above.")
            else:
                st.success("  Download complete!")

                zip_path = os.path.join(temp_dir, "papers.zip")
                with ZipFile(zip_path, "w") as zipf:
                    for root, _, files in os.walk(output_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, output_dir)
                            zipf.write(file_path, arcname)

                with open(zip_path, "rb") as f:
                    st.download_button("  Download ZIP", f, file_name="papers.zip")

    # ----------------------------------------------------------------------
    # OFFLINE MODE
    # ----------------------------------------------------------------------
    else:
	folder_path = st.text_input("  Enter path to existing downloaded folder:")

        if folder_path and os.path.exists(folder_path):
            st.session_state['output_dir'] = folder_path
            st.success(f"  Folder selected: {folder_path}")
        else:
            st.info("Provide a valid folder path with previously downloaded data.")


# ============================================================
# TAB 2: DATATABLES (amilib)
# ============================================================
with tab2:
    st.header("  Downloaded Data Summary Table (amilib)")

    if 'output_dir' not in st.session_state:
        st.warning("⚠️ Please complete Tab 1 first.")
    else:
	output_dir = st.session_state['output_dir']
        if st.button("▶️ Run amilib Summary Table"):
            st.info("  Running amilib DataTables...")
            amilib_cmd = ["amilib", "HTML", "--operation", "DATATABLES", "--indir", output_dir]
            with st.spinner("⚙️ Processing..."):
                proc = subprocess.run(amilib_cmd, capture_output=False, text=False)
            st.text(proc.stdout)
            if proc.stderr:
                st.text(proc.stderr)
            if proc.returncode == 0:
                st.success("  amilib processing complete!")

                table_dir = os.path.join(output_dir, "tables")
                if os.path.exists(table_dir):
                    for root, _, files in os.walk(table_dir):
                        for file in files:
                            if file.endswith(".csv"):
                                file_path = os.path.join(root, file)
                                st.markdown(f"####   Preview: `{file}`")
                                try:
                                    df = pd.read_csv(file_path)
                                    st.dataframe(df.head(50))
                                except Exception as e:
                                    st.warning(f"Could not load {file}: {e}")

                html_path = None
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        if file == "datatables.html":
                            html_path = os.path.join(root, file)
                            break
                    if html_path: break

                if html_path and os.path.exists(html_path):
                    st.markdown("###   DataTables Preview")
                    with open(html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    components.html(html_content, height=600, scrolling=True)
                    with open(html_path, 'rb') as f:
                        st.download_button("⬇️ Download DataTables HTML", f, file_name="datatables.html")
                else:
                    st.info("ℹ️ No datatables.html found.")
            else:
                st.error("  amilib failed. Check logs above.")

# ============================================================
# TAB 3: NER (docanalysis)

# ============================================================
with tab3:
    st.header("  Named Entity Recognition with Wikidata Linking")

    if 'output_dir' not in st.session_state:
        st.warning("⚠️ Please run the download in Tab 1 first.")
    else:
	output_dir = st.session_state['output_dir']

        available_dicts = [
            "EO_ACTIVITY", "EO_COMPOUND", "EO_EXTRACTION", "EO_PLANT",
            "EO_PLANT_PART", "PLANT_GENUS", "EO_TARGET", "COUNTRY",
            "DISEASE", "DRUG", "ORGANIZATION"
        ]

	selected_dicts = st.multiselect(
            "  Select Dictionaries",
            available_dicts,
            default=["EO_PLANT", "EO_COMPOUND"]
        )

	sections = st.multiselect(
            "  Sections to Analyze",
            ["ALL", "ACK", "AFF", "AUT", "CON", "DIS", "ETH", "FIG", "INT", "KEY", "MET", "RES", "TAB", "TIL"],
            default=["ALL"]
        )

	# --- Cache for faster lookup ---
        wikidata_cache = {}

        def fetch_wikidata_ids(entity_text, retries=3):
            """
            Fetch Wikidata IDs for one or more comma-separated entity names.
            Returns all QIDs separated by commas.
            """
            if pd.isna(entity_text) or not str(entity_text).strip():
                return None

            names = [n.strip() for n in str(entity_text).split(",") if n.strip()]
            url = "https://www.wikidata.org/w/api.php"
            headers = {"User-Agent": "PhytochemistryResearchBot/1.0 (mailto:example@domain.com)"}
            qids = []

            for name in names:
                # Use cache if already fetched
                if name in wikidata_cache:
                    if wikidata_cache[name]:
                        qids.append(wikidata_cache[name])
                    continue

                params = {
                    "action": "wbsearchentities",
                    "format": "json",
                    "language": "en",
                    "limit": 1,
                    "search": name
                }

                found_qid = None
                for attempt in range(retries):
                    try:
                        resp = requests.get(url, params=params, headers=headers, timeout=10)
                        if resp.status_code != 200:
                            time.sleep(1)
                            continue
                        data = resp.json()
                        if "search" in data and len(data["search"]) > 0:
                            found_qid = data["search"][0]["id"]
                            break
                    except Exception as e:
                        print(f"⚠️ Error fetching {name}: {e}")
                        time.sleep(1)

                wikidata_cache[name] = found_qid
                if found_qid:
                    qids.append(found_qid)
                time.sleep(0.3)  # avoid overloading the API

            return ", ".join(qids) if qids else None


        def run_docanalysis(dictionary):
            """Run docanalysis command for the selected dictionary."""
            cmd = [
                "docanalysis",
                "--project_name", output_dir,
                "--make_section",
                "--dictionary", dictionary,
                "--output", os.path.join(output_dir, f"entities_{dictionary}.csv"),
                "--make_json", os.path.join(output_dir, f"entities_{dictionary}.json")
            ]
            if sections:
                cmd += ["--search_section"] + sections
            return subprocess.run(cmd, capture_output=False, text=True)


        if st.button("  Run NER + Wikidata Linking"):
            outputs = []

            for d in selected_dicts:
                st.info(f"  Processing dictionary: **{d}** ...")
                result = run_docanalysis(d)
                st.text(result.stdout)
                if result.stderr:
                    st.text(result.stderr)

                if result.returncode == 0:
                    st.success(f"  Completed: {d}")
                    csv_path = os.path.join(output_dir, f"entities_{d}.csv")

                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)

                        if "0" in df.columns:
                            st.write("  Fetching Wikidata IDs (please wait)...")

                            progress_bar = st.progress(0)
                            wikidata_ids = []
                            total = len(df)

                            for i, val in enumerate(df["0"]):
                                qid = fetch_wikidata_ids(val)
                                wikidata_ids.append(qid)
                                progress_bar.progress((i + 1) / total)

                            df["Wikidata_ID"] = wikidata_ids
                            st.success("  Wikidata mapping completed.")
                        else:
                            st.warning(f"⚠️ Column '0' not found in {csv_path}. Skipping Wikidata mapping.")

                        out_path = os.path.join(output_dir, f"entities_{d}_wikidata.csv")
                        df.to_csv(out_path, index=False)

                        st.dataframe(df.head(50))

                        with open(out_path, "rb") as f:
                            st.download_button(
                                f"⬇️ Download CSV with Wikidata IDs ({d})",
                                f,
                                file_name=f"entities_{d}_wikidata.csv"
                            )

                        outputs.append(out_path)
                else:
                    st.error(f"  Failed to process {d}")

            if outputs:
                st.session_state['ner_outputs'] = outputs
                st.success("  NER + Wikidata CSVs ready for visualization in Tab 4.")

# ============================================================
# TAB 4: CYTOSCAPE NETWORK
# ============================================================
import os, re, json

with tab4:
    st.header("  Network Visualization")
    st.markdown("Visualize entity relationships and click nodes to open Wikidata pages instantly.")

    # --- Define color map globally inside tab4 (so it's accessible everywhere) ---
    color_map = {
        "EO_PLANT": "#2ca02c",
        "EO_COMPOUND": "#d62728",
        "EO_ACTIVITY": "#9467bd",
        "EO_EXTRACTION": "#ff7f0e",
        "EO_PLANT_PART": "#8c564b",
        "PLANT_GENUS": "#17becf",
        "EO_TARGET": "#bcbd22",
        "COUNTRY": "#e377c2",
        "DISEASE": "#7f7f7f",
        "DRUG": "#1f77b4",
        "ORGANIZATION": "#17a589",
        "FILE": "#0055ff"
    }

    if "ner_outputs" not in st.session_state:
        st.warning("⚠️ Please run the NER tab first to generate entity CSVs.")
    else:
	ner_files = st.session_state["ner_outputs"]

        st.subheader("  Available Entity CSVs")
        st.write(ner_files)

        selected_csvs = st.multiselect(
            "Select CSVs for Network Creation",
            ner_files,
            default=ner_files
        )

	layout_choice = st.selectbox(
            "Choose Layout for Visualization",
            ["cose", "circle", "grid", "breadthfirst", "concentric"],
            index=0
        )

	# --- Build Network ---
        if st.button("  Build Network"):
            all_edges = []
            wikidata_map = {}

            for csv_file in selected_csvs:
                try:
                    df = pd.read_csv(csv_file)
                    dict_name = next((key for key in color_map.keys() if key in csv_file), "Unknown")

                    if "file_path" not in df.columns or "0" not in df.columns:
                        st.warning(f"⚠️ Skipped {csv_file} (missing required columns).")
                        continue

                    for _, row in df.iterrows():
                        file_path = str(row["file_path"])
                        match = re.search(r"(PMC\d+)", file_path)
                        source = match.group(1) if match else os.path.basename(file_path)

                        targets = str(row["0"]).split(",")
                        weight = int(row["weight_0"]) if "weight_0" in df.columns else 1

                        if "Wikidata_ID" in df.columns:
                            wikidata_id = str(row["Wikidata_ID"]).split(",")[0].strip()
                            if wikidata_id and wikidata_id.lower() != "nan":
                                wikidata_map[str(row["0"]).strip()] = wikidata_id

                        for target in targets:
                            t = target.strip()
                            if t:
                                all_edges.append((source, t, weight, dict_name))
                except Exception as e:
                    st.error(f"Error reading {csv_file}: {e}")

            if not all_edges:
                st.error("No edges created — check your CSVs.")
            else:
                G = nx.Graph()
                for s, t, w, d in all_edges:
                    G.add_edge(s, t, weight=w, dict_name=d)

                # --- Nodes ---
                nodes = []
                for n in G.nodes():
                    if n.startswith("PMC"):
                        node_type = "FILE"
                        color = color_map["FILE"]
                    else:
                        edge_dicts = [G[s][n]['dict_name'] for s in G.neighbors(n) if 'dict_name' in G[s][n]]
                        node_type = edge_dicts[0] if edge_dicts else "Unknown"
                        color = color_map.get(node_type, "#cccccc")

                    wikidata_id = wikidata_map.get(n)
                    wikidata_url = f"https://www.wikidata.org/wiki/{wikidata_id}" if wikidata_id else None

                    nodes.append({
                        "data": {
                            "id": n,
                            "label": n,
                            "type": node_type,
                            "color": color,
                            "url": wikidata_url
                        }
                    })

                edges_cyto = [
                    {"data": {"source": s, "target": t, "weight": d["weight"]}}
                    for s, t, d in G.edges(data=True)
                ]

                stylesheet = [
                    {"selector": "node", "style": {
                        "label": "data(label)",
                        "background-color": "data(color)",
                        "font-size": "10px",
                        "text-valign": "center",
                        "color": "#000000",
                        "cursor": "pointer"
                    }},
                    {"selector": "edge", "style": {"width": 2, "curve-style": "bezier"}}
                ]

                st.session_state["network_elements"] = nodes + edges_cyto
                st.session_state["network_stylesheet"] = stylesheet
                st.session_state["network_layout"] = {"name": layout_choice}

                st.success(f"  Network created with {len(G.nodes())} nodes and {len(G.edges())} edges.")

    # --- Interactive Graph ---
    if "network_elements" in st.session_state:
        st.markdown("###   Interactive Graph (click node to open Wikidata)")

        elements_json = json.dumps(st.session_state["network_elements"])
        stylesheet_json = json.dumps(st.session_state["network_stylesheet"])
        layout_json = json.dumps(st.session_state["network_layout"])

        st.components.v1.html(f"""
        <html>
	<head>
          <script src="https://unpkg.com/cytoscape@3.21.2/dist/cytoscape.min.js"></script>
          <style>
            #message-box {{
              position: fixed;
              top: 20px;
              right: 20px;
              background-color: #ffefc1;
              color: #333;
              padding: 10px 15px;
              border-radius: 8px;
              font-family: sans-serif;
              box-shadow: 0px 0px 8px rgba(0,0,0,0.2);
              display: none;
              z-index: 9999;
            }}
          </style>
        </head>
        <body>
          <div id="cy" style="width:100%; height:700px; border:1px solid #ccc;"></div>
          <div id="message-box">⚠️ Wikidata page not available for this entity.</div>

          <script>
            const cy = cytoscape({{
              container: document.getElementById('cy'),
              elements: {elements_json},
              style: {stylesheet_json},
              layout: {layout_json}
            }});

            const messageBox = document.getElementById('message-box');

            function showMessage(msg) {{
              messageBox.innerText = msg;
              messageBox.style.display = 'block';
              setTimeout(() => {{
                messageBox.style.display = 'none';
              }}, 3000);
            }}

            cy.on('tap', 'node', function(evt) {{
              const node = evt.target;
              const url = node.data('url');
              const label = node.data('label');

              if (url) {{
                window.open(url, '_blank');
              }} else {{
                showMessage(`⚠️ Wikidata page not available for: ${{label}}`);
              }}
            }});
          </script>
        </body>
        </html>
        """, height=750)

        # ---   Node Color Legend ---



