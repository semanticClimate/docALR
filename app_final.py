# ============================================
# 🧪 Project Setup & Run Instructions
# ============================================
# 1️⃣ Create a new Conda environment with Python 3.11
#     conda create -n phytochem python=3.11 -y
#
# 2️⃣ Activate the environment
#     conda activate phytochem
#
# 3️⃣ Install all required libraries
#     pip install -r requirements.txt
#
# 4️⃣ Run the Streamlit app
#     streamlit run app_final.py




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

# ============================================================
# AUTO-INSTALL REQUIRED LIBRARIES (only when missing)
# ============================================================
REQUIRED_LIBS = ["streamlit", "pandas", "networkx", "st_cytoscape"]

def ensure_installed(package):
    try:
        __import__(package)
    except ImportError:
        st.info(f"📦 Installing missing package: {package}")
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=False)

for lib in REQUIRED_LIBS:
    ensure_installed(lib)

# ============================================================
# PAGE SETTINGS
# ============================================================
st.set_page_config(page_title="Essential Oil Knowledge Graph", layout="wide")

st.markdown(
    "<h1 style='color:blue; font-style:italic; font-size:40px; margin: 0; padding-left: 0px; text-align: left;'>Automated Literature Review</h1>",
    unsafe_allow_html=True
)

# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Data Download", "📊 DataTables", "🧠 NER", "🌐 Cytoscape Network"
])

# Sidebar inputs
query = st.sidebar.text_input("Search query", "Plant metabolites")
limit = st.sidebar.number_input("Max papers", 1, 1000, 10)

# ============================================================
# TAB 1: DATA DOWNLOAD (PYGETPAPERS)
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
        st.markdown("<h2 style='padding-top: 15px;'>Download Papers with Pygetpapers</h2>", unsafe_allow_html=True)

    download_options = st.multiselect(
        "📄 Choose formats to download",
        options=["PDF", "XML", "HTML"],
        default=["PDF", "XML"]
    )

    api = st.selectbox("API", [
        "europe_pmc", "crossref", "arxiv", "biorxiv", "medrxiv", "rxivist", "openalex"
    ])

    use_start = st.checkbox("Start date filter")
    start_date = st.date_input("Start date", value=date(2000, 1, 1)) if use_start else None

    use_end = st.checkbox("End date filter")
    end_date = st.date_input("End date", value=date(2023, 12, 31)) if use_end else None

    if st.button("🚀 Download & Process"):
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
        if "PDF" in download_options: cmd.append("-p")
        if "XML" in download_options: cmd.append("-x")
        if "HTML" in download_options: cmd.append("--makehtml")
        if start_date: cmd.extend(["--startdate", start_date.strftime("%Y-%m-%d")])
        if end_date: cmd.extend(["--enddate", end_date.strftime("%Y-%m-%d")])

        st.info("📥 Downloading papers using Pygetpapers... Please wait...")
        process = subprocess.run(cmd, capture_output=True, text=True)

        # Safe text output (fixes regex issue)
        if process.stdout:
            st.text(process.stdout)
        if process.stderr:
            st.text(process.stderr)

        if process.returncode != 0:
            st.error("❌ Download failed.")
        else:
            st.success("✅ Download complete!")

            zip_path = os.path.join(temp_dir, "papers.zip")
            with ZipFile(zip_path, "w") as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)

            with open(zip_path, "rb") as f:
                st.download_button("📦 Download ZIP", f, file_name="papers.zip")

# ============================================================
# TAB 2: DATATABLES (AMILib)
# ============================================================
with tab2:
    st.header("📊 Downloaded Data Summary Table (AMILib)")

    if 'output_dir' not in st.session_state:
        st.warning("⚠️ Please run the download in Tab 1 first.")
    else:
        output_dir = st.session_state['output_dir']
        
        if st.button("▶️ Run AMIlib Summary Table"):
            st.info("🔍 Running AMIlib DataTables...")

            amilib_cmd = [
                "amilib", "HTML",
                "--operation", "DATATABLES",
                "--indir", output_dir
            ]

            with st.spinner("⚙️ Processing with AMIlib..."):
                amilib_proc = subprocess.run(amilib_cmd, capture_output=True, text=True)

            # Safe display
            st.text(amilib_proc.stdout)
            if amilib_proc.stderr:
                st.text(amilib_proc.stderr)

            if amilib_proc.returncode == 0:
                st.success("✅ AMIlib processing complete!")

                table_dir = os.path.join(output_dir, "tables")
                if os.path.exists(table_dir):
                    for root, _, files in os.walk(table_dir):
                        for file in files:
                            if file.endswith(".csv"):
                                file_path = os.path.join(root, file)
                                st.markdown(f"#### 📄 Preview: `{file}`")
                                try:
                                    df = pd.read_csv(file_path)
                                    st.dataframe(df.head(50))
                                except Exception as e:
                                    st.warning(f"Could not load {file}: {e}")

                html_file_path = None
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        if file == "datatables.html":
                            html_file_path = os.path.join(root, file)
                            break
                    if html_file_path:
                        break

                if html_file_path and os.path.exists(html_file_path):
                    st.markdown("### 🌐 DataTables Preview")
                    with open(html_file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    components.html(html_content, height=600, scrolling=True)

                    with open(html_file_path, 'rb') as f:
                        st.download_button("⬇️ Download DataTables HTML", f, file_name="datatables.html", mime="text/html")
                else:
                    st.info("ℹ️ No `datatables.html` file found.")
            else:
                st.error("❌ AMIlib processing failed.")

# ============================================================
# TAB 3: NER (Docanalysis)
# ============================================================
with tab3:
    st.header("🧠 Named Entity Recognition")

    if 'output_dir' not in st.session_state:
        st.warning("⚠️ Please run the download in Tab 1 first.")
    else:
        output_dir = st.session_state['output_dir']

        available_dictionaries = [
            "EO_ACTIVITY", "EO_COMPOUND", "EO_EXTRACTION", "EO_PLANT",
            "EO_PLANT_PART", "PLANT_GENUS", "EO_TARGET", "COUNTRY",
            "DISEASE", "DRUG", "ORGANIZATION"
        ]

        selected_dictionaries = st.multiselect(
            "📚 Select Dictionaries to Process",
            options=available_dictionaries,
            default=["EO_PLANT", "EO_COMPOUND"]
        )

        search_sections = st.multiselect(
            "📄 Sections to Search",
            ["ALL", "ACK", "AFF", "AUT", "CON", "DIS", "ETH", "FIG", "INT", "KEY", "MET", "RES", "TAB", "TIL"],
            default=["ALL"]
        )

        def run_docanalysis_for_dict(output_dir, dictionary, args_list):
            cmd = [
                "docanalysis",
                "--project_name", output_dir,
                "--make_section",
                "--dictionary", dictionary,
                "--output", os.path.join(output_dir, f"entities_{dictionary}.csv"),
                "--make_json", os.path.join(output_dir, f"entities_{dictionary}.json")
            ] + args_list

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result

        if st.button("🔍 Run NER for Each Dictionary"):
            if not selected_dictionaries:
                st.warning("Please select at least one dictionary.")
            else:
                args_list = []
                if search_sections:
                    args_list += ["--search_section"] + search_sections

                generated_files = []
                for dictionary in selected_dictionaries:
                    st.info(f"🔎 Processing dictionary: `{dictionary}`...")
                    result = run_docanalysis_for_dict(output_dir, dictionary, args_list)

                    st.text(result.stdout)
                    if result.stderr:
                        st.text(result.stderr)

                    if result.returncode != 0:
                        st.error(f"❌ NER failed for `{dictionary}`.")
                        continue
                    else:
                        st.success(f"✅ NER completed for `{dictionary}`.")

                    csv_path = os.path.join(output_dir, f"entities_{dictionary}.csv")
                    if os.path.exists(csv_path):
                        generated_files.append(csv_path)
                        try:
                            df = pd.read_csv(csv_path)
                            st.markdown(f"### 📄 Preview for `{dictionary}`")
                            st.dataframe(df.head(50))
                            with open(csv_path, "rb") as f:
                                st.download_button(f"⬇️ Download CSV ({dictionary})", f, file_name=f"entities_{dictionary}.csv")
                        except Exception as e:
                            st.warning(f"⚠️ Error loading {dictionary}: {e}")

                if generated_files:
                    st.session_state['ner_outputs'] = generated_files
                    st.success("✅ NER outputs saved for Cytoscape tab!")

# ============================================================
# TAB 4: CYTOSCAPE NETWORK
# ============================================================
with tab4:
    st.header("🌐 Cytoscape Network")
    st.markdown("⚠️ Both EO_PLANT and EO_COMPOUND CSVs are required to visualize relationships.")

    if "ner_outputs" not in st.session_state:
        st.warning("⚠️ Please run the NER tab first to generate entity CSVs.")
    else:
        ner_files = st.session_state["ner_outputs"]

        st.subheader("📁 Available NER CSVs")
        st.write(ner_files)

        selected_csvs = st.multiselect(
            "Select CSVs for Network Creation",
            ner_files,
            default=ner_files
        )

        layout_choice = st.selectbox(
            "🧩 Choose Layout for Visualization",
            ["cose", "circle", "grid", "breadthfirst", "concentric"],
            index=0
        )

        if st.button("🔗 Build Network"):
            all_edges = []

            for csv_file in selected_csvs:
                try:
                    df = pd.read_csv(csv_file)

                    if "file_path" not in df.columns or "0" not in df.columns:
                        st.warning(f"⚠️ Skipped {csv_file} (missing required columns).")
                        continue

                    for _, row in df.iterrows():
                        file_path = str(row["file_path"])
                        match = re.search(r"(PMC\d+)", file_path)
                        source = match.group(1) if match else os.path.basename(file_path)

                        targets = str(row["0"]).split(",")
                        weight = int(row["weight_0"]) if "weight_0" in df.columns else 1

                        for target in targets:
                            t = target.strip()
                            if t:
                                all_edges.append((source, t, weight))
                except Exception as e:
                    st.error(f"Error reading {csv_file}: {e}")

            if not all_edges:
                st.error("❌ No edges created — check your CSVs.")
            else:
                G = nx.Graph()
                for s, t, w in all_edges:
                    G.add_edge(s, t, weight=w)

                nodes = []
                for n in G.nodes():
                    node_type = "File" if n.startswith("PMC") else "Entity"
                    color = "#1f77b4" if node_type == "File" else "#2ca02c"
                    nodes.append({
                        "data": {"id": n, "label": n, "type": node_type, "color": color}
                    })

                edges_cyto = [
                    {"data": {"source": s, "target": t, "weight": d["weight"]}}
                    for s, t, d in G.edges(data=True)
                ]

                stylesheet = [
                    {"selector": "node", "style": {
                        "label": "data(id)",
                        "background-color": "data(color)",
                        "font-size": "10px",
                        "text-valign": "center"}},
                    {"selector": "edge", "style": {"width": 3, "curve-style": "bezier"}},
                ]

                st.session_state["network_elements"] = nodes + edges_cyto
                st.session_state["network_stylesheet"] = stylesheet
                st.session_state["network_layout"] = {"name": layout_choice}

                st.success(f"✅ Network created with {len(G.nodes())} nodes and {len(G.edges())} edges.")

        if "network_elements" in st.session_state:
            st.markdown("### 🌿 Interactive Graph")

            cytoscape(
                elements=st.session_state["network_elements"],
                stylesheet=st.session_state["network_stylesheet"],
                height="700px",
                width="100%",
                key="eo_network",
                layout=st.session_state["network_layout"]
            )
