# src/dashboard/app.py
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from PIL import Image
import io
import base64

# API endpoint
API_ENDPOINT = "http://localhost:8000"


def main():
    st.title("Multimodal AI System Dashboard")

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Upload & Process", "Knowledge Graph", "Analytics"]
    )

    if page == "Upload & Process":
        upload_and_process()
    elif page == "Knowledge Graph":
        knowledge_graph_explorer()
    elif page == "Analytics":
        analytics()


def upload_and_process():
    st.header("Upload & Process Content")

    # Create tabs for different input types
    tab1, tab2, tab3, tab4 = st.tabs(["Text", "Image", "Video", "Multimodal"])

    with tab1:
        process_text()

    with tab2:
        process_image()

    with tab3:
        process_video()

    with tab4:
        process_multimodal()


def process_text():
    st.subheader("Process Text")

    text_input = st.text_area("Enter text to process", height=200)

    if st.button("Process Text"):
        if text_input:
            # Call API
            with st.spinner("Processing text..."):
                response = requests.post(
                    f"{API_ENDPOINT}/process/text",
                    json={"text": text_input}
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display results
                    st.success("Text processed successfully!")

                    # Classification result
                    st.subheader("Classification")
                    st.write(f"Class: {result['classification']['class_name']}")
                    st.write(f"Confidence: {result['classification']['confidence']:.2f}")

                    # Entities
                    st.subheader("Entities")
                    entities_df = pd.DataFrame(result["entities"])
                    st.dataframe(entities_df)

                    # Relations
                    if result["relations"]:
                        st.subheader("Relations")
                        relations_df = pd.DataFrame(result["relations"])
                        st.dataframe(relations_df)

                    # Summary
                    st.subheader("Summary")
                    st.write(result["summary"])
                else:
                    st.error(f"Error processing text: {response.text}")
        else:
            st.warning("Please enter some text to process.")


def process_image():
    st.subheader("Process Image")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # src/dashboard/app.py (continued)
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Process Image"):
            # Call API
            with st.spinner("Processing image..."):
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{API_ENDPOINT}/process/image", files=files)

                if response.status_code == 200:
                    result = response.json()

                    # Display results
                    st.success("Image processed successfully!")

                    # Classification result
                    st.subheader("Classification")
                    st.write(f"Class: {result['classification']['class_name']}")
                    st.write(f"Confidence: {result['classification']['confidence']:.2f}")

                    # Objects detected
                    st.subheader("Objects Detected")
                    objects_df = pd.DataFrame(result["objects"])
                    st.dataframe(objects_df)

                    # Visualize bounding boxes
                    if len(result["objects"]) > 0:
                        fig = visualize_bounding_boxes(image, result["objects"])
                        st.plotly_chart(fig)
                else:
                    st.error(f"Error processing image: {response.text}")


def process_video():
    st.subheader("Process Video")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Display the uploaded video
        video_bytes = uploaded_file.read()
        st.video(video_bytes)

        if st.button("Process Video"):
            # Call API
            with st.spinner("Processing video... This may take a while."):
                files = {"file": uploaded_file}
                response = requests.post(f"{API_ENDPOINT}/process/video", files=files)

                if response.status_code == 200:
                    result = response.json()

                    # Display results
                    st.success("Video processed successfully!")

                    # Video highlights
                    if "summary" in result and "video_highlights" in result["summary"]:
                        st.subheader("Video Highlights")
                        highlight_url = f"{API_ENDPOINT}{result['summary']['video_highlights']}"
                        st.video(highlight_url)
                else:
                    st.error(f"Error processing video: {response.text}")


def process_multimodal():
    st.subheader("Process Multimodal Content")

    # Text input
    text_input = st.text_area("Enter text (optional)", height=100)

    # Image upload
    image_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])

    # Video upload
    video_file = st.file_uploader("Upload a video (optional)", type=["mp4", "avi", "mov"])

    # Audio upload
    audio_file = st.file_uploader("Upload an audio file (optional)", type=["mp3", "wav"])

    if st.button("Process Multimodal Content"):
        if text_input or image_file or video_file or audio_file:
            # Call API
            with st.spinner("Processing multimodal content..."):
                files = {}
                data = {}

                if text_input:
                    data["text"] = text_input

                if image_file:
                    files["image"] = image_file

                if video_file:
                    files["video"] = video_file

                if audio_file:
                    files["audio"] = audio_file

                response = requests.post(
                    f"{API_ENDPOINT}/process/multimodal",
                    data=data,
                    files=files
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display results
                    st.success("Multimodal content processed successfully!")

                    # Classification result
                    if "classification" in result:
                        st.subheader("Classification")
                        st.write(f"Class: {result['classification']['class_name']}")
                        st.write(f"Confidence: {result['classification']['confidence']:.2f}")

                    # Text results
                    if "text" in result:
                        st.subheader("Text Analysis")
                        if "entities" in result["text"]:
                            st.write("Entities:")
                            entities_df = pd.DataFrame(result["text"]["entities"])
                            st.dataframe(entities_df)

                    # Image results
                    if "image" in result:
                        st.subheader("Image Analysis")
                        if "objects" in result["image"]:
                            st.write("Objects:")
                            objects_df = pd.DataFrame(result["image"]["objects"])
                            st.dataframe(objects_df)

                    # Multimodal summary
                    if "multimodal_summary" in result:
                        st.subheader("Multimodal Summary")

                        if "video_highlights" in result["multimodal_summary"]:
                            st.write("Video Highlights:")
                            highlight_url = f"{API_ENDPOINT}{result['multimodal_summary']['video_highlights']}"
                            st.video(highlight_url)
                else:
                    st.error(f"Error processing multimodal content: {response.text}")
        else:
            st.warning("Please provide at least one type of content to process.")


def knowledge_graph_explorer():
    st.header("Knowledge Graph Explorer")

    # Cypher query input
    query = st.text_area(
        "Enter Cypher Query",
        value="MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100",
        height=100
    )

    if st.button("Execute Query"):
        # Call API
        with st.spinner("Executing query..."):
            response = requests.get(
                f"{API_ENDPOINT}/knowledge-graph/query",
                params={"query": query}
            )

            if response.status_code == 200:
                result = response.json()

                # Display results
                st.success("Query executed successfully!")

                # Convert results to a graph
                G = create_graph_from_results(result["results"])

                # Visualize graph
                fig = visualize_graph(G)
                st.plotly_chart(fig)

                # Show raw results
                st.subheader("Raw Results")
                st.json(result)
            else:
                st.error(f"Error executing query: {response.text}")

    # Pre-defined queries
    st.subheader("Pre-defined Queries")

    if st.button("Show all Person entities"):
        execute_predefined_query("MATCH (n:Person) RETURN n LIMIT 100")

    if st.button("Show all Organization entities"):
        execute_predefined_query("MATCH (n:Organization) RETURN n LIMIT 100")

    if st.button("Show connections between people"):
        execute_predefined_query(
            "MATCH (p1:Person)-[r]-(p2:Person) RETURN p1, r, p2 LIMIT 100"
        )


def analytics():
    st.header("Analytics Dashboard")

    # Tabs for different analytics
    tab1, tab2, tab3 = st.tabs(["Content Statistics", "Entity Distribution", "Classification Performance"])

    with tab1:
        content_statistics()

    with tab2:
        entity_distribution()

    with tab3:
        classification_performance()


def content_statistics():
    st.subheader("Content Statistics")

    # Fake data for demonstration
    content_types = ["Text", "Image", "Video", "Audio", "Multimodal"]
    content_counts = [245, 187, 93, 76, 134]

    # Create bar chart
    fig = px.bar(
        x=content_types,
        y=content_counts,
        title="Content Distribution by Type",
        labels={"x": "Content Type", "y": "Count"}
    )

    st.plotly_chart(fig)

    # Processing time metrics
    st.subheader("Average Processing Time (ms)")

    processing_times = {
        "Text": 325,
        "Image": 780,
        "Video": 2450,
        "Audio": 520,
        "Multimodal": 3100
    }

    fig = px.bar(
        x=list(processing_times.keys()),
        y=list(processing_times.values()),
        title="Average Processing Time by Content Type",
        labels={"x": "Content Type", "y": "Time (ms)"}
    )

    st.plotly_chart(fig)


def entity_distribution():
    st.subheader("Entity Distribution")

    # Fake data for demonstration
    entity_types = ["Person", "Organization", "Location", "Date", "Event", "Product"]
    entity_counts = [572, 389, 426, 287, 134, 298]

    # Create pie chart
    fig = px.pie(
        names=entity_types,
        values=entity_counts,
        title="Entity Type Distribution"
    )

    st.plotly_chart(fig)

    # Entity connections
    st.subheader("Entity Connections")

    # Create sample graph
    G = nx.DiGraph()

    # Add nodes
    G.add_node("Person", count=572)
    G.add_node("Organization", count=389)
    G.add_node("Location", count=426)
    G.add_node("Date", count=287)
    G.add_node("Event", count=134)
    G.add_node("Product", count=298)

    # Add edges
    G.add_edge("Person", "Organization", weight=245)
    G.add_edge("Person", "Location", weight=198)
    G.add_edge("Organization", "Location", weight=176)
    G.add_edge("Event", "Date", weight=124)
    G.add_edge("Person", "Event", weight=98)
    G.add_edge("Organization", "Event", weight=87)
    G.add_edge("Product", "Organization", weight=143)
    G.add_edge("Person", "Product", weight=112)

    # Visualize graph
    fig = visualize_graph(G)
    st.plotly_chart(fig)


def classification_performance():
    st.subheader("Classification Performance")

    # Fake data for demonstration
    categories = ["Politics", "Sports", "Entertainment", "Technology", "Business"]
    precision = [0.92, 0.87, 0.89, 0.94, 0.85]
    recall = [0.88, 0.85, 0.91, 0.90, 0.83]
    f1_score = [0.90, 0.86, 0.90, 0.92, 0.84]

    # Create grouped bar chart
    df = pd.DataFrame({
        "Category": categories * 3,
        "Metric Value": precision + recall + f1_score,
        "Metric": ["Precision"] * 5 + ["Recall"] * 5 + ["F1 Score"] * 5
    })

    fig = px.bar(
        df,
        x="Category",
        y="Metric Value",
        color="Metric",
        barmode="group",
        title="Classification Performance Metrics by Category",
        labels={"Metric Value": "Score"}
    )

    st.plotly_chart(fig)

    # Confusion matrix
    st.subheader("Confusion Matrix")

    # Fake confusion matrix
    confusion_matrix = [
        [112, 8, 5, 3, 2],
        [6, 98, 7, 3, 4],
        [4, 5, 103, 2, 6],
        [2, 1, 3, 109, 5],
        [3, 4, 5, 4, 94]
    ]

    fig = px.imshow(
        confusion_matrix,
        x=categories,
        y=categories,
        color_continuous_scale="blues",
        title="Confusion Matrix",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"}
    )

    st.plotly_chart(fig)


def visualize_bounding_boxes(image, objects):
    """Visualize bounding boxes on an image"""
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Create figure
    fig = go.Figure()

    # Add image
    fig.add_trace(
        go.Image(z=img_array)
    )

    # Add bounding boxes
    for obj in objects:
        box = obj["box"]
        x0, y0, x1, y1 = box

        # Add rectangle
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="red", width=2),
            name=f"{obj['label']} ({obj['score']:.2f})"
        )

        # Add label
        fig.add_annotation(
            x=x0,
            y=y0,
            text=f"{obj['label']} ({obj['score']:.2f})",
            showarrow=False,
            font=dict(color="white", size=10),
            bgcolor="rgba(255, 0, 0, 0.5)",
            bordercolor="red",
            borderwidth=1
        )

    # Update layout
    fig.update_layout(
        height=600,
        width=800,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig


def create_graph_from_results(results):
    """Create a NetworkX graph from Neo4j query results"""
    G = nx.DiGraph()

    for result in results:
        # Add nodes and edges based on query result
        # This will need to be adapted based on your actual Neo4j data structure
        for key, value in result.items():
            if key.endswith("_id"):
                # Skip ID fields
                continue

            if isinstance(value, dict) and "type" in value and "properties" in value:
                # Node
                node_id = value.get("id", str(hash(str(value))))
                node_label = value.get("labels", [value["type"]])[0]

                G.add_node(node_id, label=node_label, **value["properties"])
            elif isinstance(value, dict) and "type" in value and "start" in value and "end" in value:
                # Relationship
                source_id = value["start"]
                target_id = value["end"]
                rel_type = value["type"]

                G.add_edge(source_id, target_id, type=rel_type)

    return G


def visualize_graph(G):
    """Visualize a NetworkX graph using Plotly"""
    # Use spring layout for node positions
    pos = nx.spring_layout(G, seed=42)

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []

    for node, attrs in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Node label
        if "label" in attrs:
            label = attrs["label"]
        else:
            label = node

        # Node size (based on 'count' attribute if available)
        if "count" in attrs:
            size = attrs["count"] / 10
        else:
            size = 10

        node_size.append(size)

        # Node text (for hover)
        text = f"Node: {label}"
        for key, value in attrs.items():
            if key != "label":
                text += f"<br>{key}: {value}"

        node_text.append(text)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=node_size,
            color="lightblue",
            line=dict(width=1, color="black")
        )
    )

    # Create edge traces
    edge_traces = []

    for edge in G.edges(data=True):
        source, target, attrs = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]

        # Edge weight (if available)
        width = attrs.get("weight", 1)

        # Edge type
        edge_type = attrs.get("type", "")

        # Create edge trace
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color="gray"),
            hoverinfo="text",
            text=f"Edge: {source} -> {target}<br>Type: {edge_type}<br>Weight: {width}"
        )

        edge_traces.append(edge_trace)

    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title="Knowledge Graph Visualization",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    return fig


def execute_predefined_query(query):
    """Execute a predefined Cypher query"""
    # Call API
    with st.spinner("Executing query..."):
        response = requests.get(
            f"{API_ENDPOINT}/knowledge-graph/query",
            params={"query": query}
        )

        if response.status_code == 200:
            result = response.json()

            # Display results
            st.success("Query executed successfully!")

            # Convert results to a graph
            G = create_graph_from_results(result["results"])

            # Visualize graph
            fig = visualize_graph(G)
            st.plotly_chart(fig)

            # Show raw results
            st.subheader("Raw Results")
            st.json(result)
        else:
            st.error(f"Error executing query: {response.text}")


if __name__ == "__main__":
    main()