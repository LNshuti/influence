import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Load data
guest_by_channel = pd.read_csv("data/table_guests_by_channel.csv").rename(columns=str.lower)
topic_summaries = pd.read_csv("data/table_topic_summaries.csv").rename(columns=str.lower)
videos_by_topic = pd.read_csv("data/videos_by_topic.csv").rename(columns=str.lower)
youtube_metadata = pd.read_csv("data/youtube_metadata.tsv", sep="\t").rename(columns=str.lower)
guest_timeline = pd.read_csv("data/guest_timeline.tsv", sep="\t").rename(columns=str.lower)

# Cleaning and preprocessing
guest_by_channel.columns = guest_by_channel.columns.str.replace(' ', '_')
topic_summaries.columns = topic_summaries.columns.str.replace(' ', '_')

# Convert date columns to datetime
youtube_metadata['video_publish_date'] = pd.to_datetime(youtube_metadata['video_publish_date'])
guest_timeline['video_publish_date'] = pd.to_datetime(guest_timeline['video_publish_date'])
videos_by_topic['video_publish_date'] = pd.to_datetime(videos_by_topic['video_publish_date'])

# Create derived metrics
guest_by_channel['avg_views_per_channel'] = guest_by_channel['views_sum'] / guest_by_channel['no_of_channels']
guest_by_channel = guest_by_channel.sort_values('views_sum', ascending=False)

# Create channel list for filtering
channel_opts = [col for col in guest_by_channel.columns if col in 
               ['adin_live', 'flagrant', 'full_send_podcast', 'impaulsive', 
                'lex_fridman', 'pbd_podcast', 'powerfuljre', 'shawn_ryan_show', 'theo_von']]

# Create topic list for filtering
topic_opts = [col for col in topic_summaries.columns if col.startswith('#')]

# Create category list for filtering
categories = sorted(guest_by_channel['category'].unique().tolist())

# ---- DASHBOARD COMPONENTS ----

# 1. Executive Summary
def executive_summary():
    # Calculate key metrics
    total_guests = len(guest_by_channel)
    total_views = guest_by_channel['views_sum'].sum()
    avg_views_per_guest = total_views / total_guests
    female_guests = guest_by_channel[guest_by_channel['is_a_woman'] == True].shape[0]
    female_pct = (female_guests / total_guests) * 100
    
    # Top performing categories by views
    category_views = guest_by_channel.groupby('category')['views_sum'].sum().sort_values(ascending=False)
    
    # Top performing channels by guest appearances
    channel_appearances = {}
    for channel in channel_opts:
        channel_appearances[channel] = guest_by_channel[guest_by_channel[channel] == 1].shape[0]
    
    # Create summary visualizations
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("Total Guest Views (M)", "Avg Views per Guest (M)", 
                       "Guest Categories", "Guest Gender Distribution")
    )
    
    # Add indicators
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=total_views / 1000000,
            number={"suffix": "M", "valueformat": ".1f"},
            title={"text": "Total Views"}
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=avg_views_per_guest / 1000000,
            number={"suffix": "M", "valueformat": ".1f"},
            title={"text": "Avg Views per Guest"}
        ),
        row=1, col=2
    )
    
    # Add category pie chart
    fig.add_trace(
        go.Pie(
            labels=category_views.index,
            values=category_views.values,
            hole=0.4,
            textinfo="label+percent"
        ),
        row=2, col=1
    )
    
    # Add gender bar chart
    fig.add_trace(
        go.Bar(
            x=["Male", "Female"],
            y=[total_guests - female_guests, female_guests],
            text=[f"{100-female_pct:.1f}%", f"{female_pct:.1f}%"],
            textposition="auto"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="Executive Dashboard - Key Performance Metrics",
        showlegend=False
    )
    
    return fig

# 2. Guest Performance Analysis
def guest_performance_analysis(top_n=20, category_filter=None, gender_filter=None):
    # Filter data based on inputs
    filtered_data = guest_by_channel.copy()
    
    if category_filter and category_filter != "All Categories":
        filtered_data = filtered_data[filtered_data['category'] == category_filter]
    
    if gender_filter == "Female":
        filtered_data = filtered_data[filtered_data['is_a_woman'] == True]
    elif gender_filter == "Male":
        filtered_data = filtered_data[filtered_data['is_a_woman'] == False]
    
    # Get top N guests
    top_guests = filtered_data.head(top_n)
    
    # Create visualization
    fig = px.bar(
        top_guests, 
        x='guest', 
        y='views_sum',
        color='category',
        hover_data=['no_of_channels', 'avg_views_per_channel'],
        labels={
            'guest': 'Guest Name',
            'views_sum': 'Total Views',
            'category': 'Guest Category',
            'no_of_channels': 'Number of Channels',
            'avg_views_per_channel': 'Avg Views per Channel'
        },
        title=f'Top {top_n} Guests by Total Views',
        height=600
    )
    
    fig.update_layout(
        xaxis_title="Guest",
        yaxis_title="Total Views",
        xaxis={'categoryorder':'total descending'},
        yaxis=dict(tickformat=".2s")
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig

# 3. Channel Comparison
def channel_comparison(selected_channels, metric="guest_count"):
    if not selected_channels:
        selected_channels = channel_opts[:3]  # Default to first 3 channels
    
    # Prepare data based on selected metric
    if metric == "guest_count":
        # Count guests per channel
        channel_data = {channel: guest_by_channel[guest_by_channel[channel] == 1].shape[0] for channel in selected_channels}
        title = "Number of Guests per Channel"
        y_label = "Guest Count"
    
    elif metric == "total_views":
        # Sum views per channel
        channel_data = {channel: guest_by_channel[guest_by_channel[channel] == 1]['views_sum'].sum() for channel in selected_channels}
        title = "Total Views per Channel"
        y_label = "Total Views"
    
    elif metric == "avg_views":
        # Average views per guest per channel
        channel_data = {channel: guest_by_channel[guest_by_channel[channel] == 1]['views_sum'].mean() for channel in selected_channels}
        title = "Average Views per Guest per Channel"
        y_label = "Average Views"
    
    elif metric == "category_diversity":
        # Category diversity per channel (number of unique categories)
        channel_data = {channel: len(guest_by_channel[guest_by_channel[channel] == 1]['category'].unique()) for channel in selected_channels}
        title = "Category Diversity per Channel"
        y_label = "Number of Unique Categories"
    
    # Create visualization
    fig = px.bar(
        x=list(channel_data.keys()),
        y=list(channel_data.values()),
        labels={'x': 'Channel', 'y': y_label},
        title=title,
        height=500
    )
    
    # Format y-axis for views
    if metric in ["total_views", "avg_views"]:
        fig.update_layout(yaxis=dict(tickformat=".2s"))
    
    return fig

# 4. Topic Trend Analysis
def topic_trend_analysis(selected_topics, time_period="all"):
    if not selected_topics:
        selected_topics = topic_opts[:3]  # Default to first 3 topics
    
    # Filter data based on time period
    filtered_data = videos_by_topic.copy()
    
    if time_period == "last_year":
        one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
        filtered_data = filtered_data[filtered_data['video_publish_date'] >= one_year_ago]
    elif time_period == "last_6_months":
        six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
        filtered_data = filtered_data[filtered_data['video_publish_date'] >= six_months_ago]
    
    # Group by month and calculate topic frequency
    filtered_data['month'] = filtered_data['video_publish_date'].dt.to_period('M')
    
    # Create dataframe for visualization
    topic_trends = []
    
    for topic in selected_topics:
        if topic in filtered_data.columns:
            monthly_data = filtered_data.groupby('month')[topic].mean().reset_index()
            monthly_data['topic'] = topic
            monthly_data['month'] = monthly_data['month'].dt.to_timestamp()
            topic_trends.append(monthly_data)
    
    if not topic_trends:
        return go.Figure().update_layout(title="No data available for selected topics")
    
    trend_df = pd.concat(topic_trends)
    
    # Create visualization
    fig = px.line(
        trend_df,
        x='month',
        y=topic,
        color='topic',
        labels={
            'month': 'Month',
            topic: 'Topic Frequency',
            'topic': 'Topic'
        },
        title='Topic Trends Over Time',
        height=500
    )
    
    return fig

# 5. Guest Category ROI Analysis
def guest_category_roi(metric="views_per_appearance"):
    # Calculate metrics by category
    category_metrics = guest_by_channel.groupby('category').agg(
        total_views=('views_sum', 'sum'),
        guest_count=('guest', 'count'),
        total_appearances=('no_of_channels', 'sum')
    ).reset_index()
    
    # Calculate derived metrics
    category_metrics['views_per_guest'] = category_metrics['total_views'] / category_metrics['guest_count']
    category_metrics['views_per_appearance'] = category_metrics['total_views'] / category_metrics['total_appearances']
    
    # Select metric for visualization
    if metric == "views_per_guest":
        y_value = 'views_per_guest'
        title = 'Views per Guest by Category'
        y_label = 'Views per Guest'
    else:  # views_per_appearance
        y_value = 'views_per_appearance'
        title = 'Views per Appearance by Category'
        y_label = 'Views per Appearance'
    
    # Create visualization
    fig = px.bar(
        category_metrics.sort_values(y_value, ascending=False),
        x='category',
        y=y_value,
        color='guest_count',
        text='guest_count',
        labels={
            'category': 'Guest Category',
            y_value: y_label,
            'guest_count': 'Number of Guests'
        },
        title=title,
        height=500
    )
    
    fig.update_layout(yaxis=dict(tickformat=".2s"))
    
    return fig

# 6. Content Strategy Recommendations
def content_strategy_recommendations(selected_topics=None):
    if not selected_topics:
        selected_topics = topic_opts[:5]  # Default to first 5 topics
    
    # Calculate engagement metrics for videos by topic
    topic_engagement = {}
    
    for topic in selected_topics:
        if topic in videos_by_topic.columns:
            # Filter videos that cover this topic
            topic_videos = videos_by_topic[videos_by_topic[topic] > 0]
            
            if not topic_videos.empty:
                # Calculate metrics
                avg_views = topic_videos['video_view_count'].mean()
                avg_likes = topic_videos['video_like_count'].mean()
                avg_comments = topic_videos['video_comment_count'].mean()
                
                # Calculate engagement rate (likes + comments) / views
                engagement_rate = (avg_likes + avg_comments) / avg_views if avg_views > 0 else 0
                
                topic_engagement[topic] = {
                    'avg_views': avg_views,
                    'avg_likes': avg_likes,
                    'avg_comments': avg_comments,
                    'engagement_rate': engagement_rate
                }
    
    # Create dataframe for visualization
    engagement_df = pd.DataFrame.from_dict(topic_engagement, orient='index').reset_index()
    engagement_df.rename(columns={'index': 'topic'}, inplace=True)
    
    if engagement_df.empty:
        return go.Figure().update_layout(title="No data available for selected topics")
    
    # Create visualization
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "scatter"}]],
        subplot_titles=("Average Views by Topic", "Engagement Analysis")
    )
    
    # Add average views bar chart
    fig.add_trace(
        go.Bar(
            x=engagement_df['topic'],
            y=engagement_df['avg_views'],
            name='Avg Views'
        ),
        row=1, col=1
    )
    
    # Add engagement scatter plot
    fig.add_trace(
        go.Scatter(
            x=engagement_df['avg_views'],
            y=engagement_df['engagement_rate'],
            mode='markers+text',
            text=engagement_df['topic'],
            textposition="top center",
            marker=dict(
                size=engagement_df['avg_comments'] / 100,  # Size based on comment count
                sizemin=10,
                sizemode='area'
            ),
            name='Engagement Rate'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        title_text="Content Strategy Analysis by Topic",
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Average Views", row=1, col=1)
    fig.update_yaxes(title_text="Engagement Rate (Likes+Comments)/Views", row=1, col=2)
    fig.update_xaxes(title_text="Topic", row=1, col=1)
    fig.update_xaxes(title_text="Average Views", row=1, col=2)
    
    return fig

# 7. Guest Timeline Analysis
def guest_timeline_analysis(selected_guest, view_type="views"):
    if not selected_guest:
        # Default to highest viewed guest
        selected_guest = guest_by_channel.iloc[0]['guest']
    
    # Filter data for selected guest
    guest_data = guest_timeline[guest_timeline['guest'] == selected_guest].copy()
    
    if guest_data.empty:
        return go.Figure().update_layout(title=f"No timeline data available for {selected_guest}")
    
    # Sort by date
    guest_data = guest_data.sort_values('video_publish_date')
    
    # Create visualization based on view type
    if view_type == "views":
        fig = px.line(
            guest_data,
            x='video_publish_date',
            y='video_view_count',
            color='channel_title',
            markers=True,
            labels={
                'video_publish_date': 'Date',
                'video_view_count': 'Views',
                'channel_title': 'Channel'
            },
            title=f'View Count Timeline for {selected_guest}',
            height=500
        )
        
        # Add average line
        avg_views = guest_data['video_view_count'].mean()
        fig.add_hline(y=avg_views, line_dash="dash", line_color="gray", 
                     annotation_text=f"Avg: {avg_views:.0f} views")
        
    else:  # cumulative
        guest_data = guest_data.sort_values('video_publish_date')
        guest_data['cumulative_views'] = guest_data['video_view_count'].cumsum()
        
        fig = px.line(
            guest_data,
            x='video_publish_date',
            y='cumulative_views',
            markers=True,
            labels={
                'video_publish_date': 'Date',
                'cumulative_views': 'Cumulative Views'
            },
            title=f'Cumulative Views for {selected_guest}',
            height=500
        )
    
    return fig

# 8. Channel Growth Analysis
def channel_growth_analysis(selected_channels):
    if not selected_channels:
        selected_channels = channel_opts[:3]  # Default to first 3 channels
    
    # Filter metadata for selected channels
    channel_data = youtube_metadata[youtube_metadata['channel_title'].str.lower().isin([ch.replace('_', ' ') for ch in selected_channels])]
    
    if channel_data.empty:
        return go.Figure().update_layout(title="No data available for selected channels")
    
    # Group by channel and month
    channel_data['month'] = channel_data['video_publish_date'].dt.to_period('M')
    monthly_stats = channel_data.groupby(['channel_title', 'month']).agg(
        avg_views=('video_view_count', 'mean'),
        video_count=('video_id', 'count')
    ).reset_index()
    
    monthly_stats['month'] = monthly_stats['month'].dt.to_timestamp()
    
    # Create visualization
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter"}, {"type": "bar"}]],
        subplot_titles=("Average Views per Video Over Time", "Monthly Video Production")
    )
    
    # Add average views line chart
    for channel in monthly_stats['channel_title'].unique():
        channel_monthly = monthly_stats[monthly_stats['channel_title'] == channel]
        
        fig.add_trace(
            go.Scatter(
                x=channel_monthly['month'],
                y=channel_monthly['avg_views'],
                mode='lines+markers',
                name=channel
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=channel_monthly['month'],
                y=channel_monthly['video_count'],
                name=channel
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=500,
        title_text="Channel Growth Analysis",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Average Views per Video", row=1, col=1)
    fig.update_yaxes(title_text="Number of Videos", row=1, col=2)
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=1, col=2)
    
    return fig

# ---- GRADIO INTERFACE ----

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 YouTube Content Strategy & Analytics Dashboard")
    gr.Markdown("### Business Intelligence for Content Strategy and Guest Selection")
    
    with gr.Tab("Executive Summary"):
        gr.Markdown("### Key Performance Indicators and Business Overview")
        exec_summary_plot = gr.Plot()
        gr.Button("Generate Executive Summary").click(fn=executive_summary, outputs=exec_summary_plot)
    
    with gr.Tab("Guest Performance"):
        gr.Markdown("### Guest Performance Analysis")
        with gr.Row():
            with gr.Column(scale=1):
                top_n = gr.Slider(minimum=5, maximum=50, value=20, step=5, label="Number of Guests")
                category_filter = gr.Dropdown(choices=["All Categories"] + categories, value="All Categories", label="Filter by Category")
                gender_filter = gr.Dropdown(choices=["All", "Male", "Female"], value="All", label="Filter by Gender")
                guest_perf_btn = gr.Button("Analyze Guest Performance")
            
            with gr.Column(scale=3):
                guest_perf_plot = gr.Plot()
        
        guest_perf_btn.click(
            fn=guest_performance_analysis,
            inputs=[top_n, category_filter, gender_filter],
            outputs=guest_perf_plot
        )
    
    with gr.Tab("Channel Analysis"):
        gr.Markdown("### Channel Comparison and Performance")
        with gr.Row():
            with gr.Column(scale=1):
                channel_select = gr.CheckboxGroup(choices=channel_opts, value=channel_opts[:3], label="Select Channels")
                metric_select = gr.Radio(
                    choices=["guest_count", "total_views", "avg_views", "category_diversity"],
                    value="total_views",
                    label="Comparison Metric"
                )
                channel_btn = gr.Button("Compare Channels")
            
            with gr.Column(scale=3):
                channel_plot = gr.Plot()
        
        channel_btn.click(
            fn=channel_comparison,
            inputs=[channel_select, metric_select],
            outputs=channel_plot
        )
    
    with gr.Tab("Topic Trends"):
        gr.Markdown("### Topic Trend Analysis")
        with gr.Row():
            with gr.Column(scale=1):
                topic_select = gr.CheckboxGroup(choices=topic_opts, value=topic_opts[:3], label="Select Topics")
                time_period = gr.Radio(
                    choices=["all", "last_year", "last_6_months"],
                    value="all",
                    label="Time Period"
                )
                topic_btn = gr.Button("Analyze Topic Trends")
            
            with gr.Column(scale=3):
                topic_plot = gr.Plot()
        
        topic_btn.click(
            fn=topic_trend_analysis,
            inputs=[topic_select, time_period],
            outputs=topic_plot
        )
    
    with gr.Tab("ROI Analysis"):
        gr.Markdown("### Return on Investment by Guest Category")
        with gr.Row():
            with gr.Column(scale=1):
                roi_metric = gr.Radio(
                    choices=["views_per_appearance", "views_per_guest"],
                    value="views_per_appearance",
                    label="ROI Metric"
                )
                roi_btn = gr.Button("Calculate ROI")
            
            with gr.Column(scale=3):
                roi_plot = gr.Plot()
        
        roi_btn.click(
            fn=guest_category_roi,
            inputs=[roi_metric],
            outputs=roi_plot
        )
    
    
    with gr.Tab("Guest Timeline"):
        gr.Markdown("### Guest Performance Timeline")
        with gr.Row():
            with gr.Column(scale=1):
                guest_select = gr.Dropdown(choices=sorted(guest_by_channel['guest'].unique().tolist()), label="Select Guest")
                timeline_type = gr.Radio(
                    choices=["views", "cumulative"],
                    value="views",
                    label="Timeline View"
                )
                timeline_btn = gr.Button("Analyze Timeline")
            
            with gr.Column(scale=3):
                timeline_plot = gr.Plot()
        
        timeline_btn.click(
            fn=guest_timeline_analysis,
            inputs=[guest_select, timeline_type],
            outputs=timeline_plot
        )
    
    with gr.Tab("Channel Growth"):
        gr.Markdown("### Channel Growth Analysis")
        with gr.Row():
            with gr.Column(scale=1):
                growth_channels = gr.CheckboxGroup(choices=channel_opts, value=channel_opts[:3], label="Select Channels")
                growth_btn = gr.Button("Analyze Growth")
            
            with gr.Column(scale=3):
                growth_plot = gr.Plot()
        
        growth_btn.click(
            fn=channel_growth_analysis,
            inputs=[growth_channels],
            outputs=growth_plot
        )

if __name__ == "__main__":
    demo.launch()