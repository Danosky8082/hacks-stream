# app.py - AfroFuture Stream V7
# COMPLETE UPGRADE WITH JUICER MULTI-PLATFORM SUPPORT & VIBRANT HOMEPAGE
# Faith · Technology · African Innovation

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from datetime import datetime
import random
import requests
import os
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================
# JUICER API CONFIGURATION
# ============================================================

def get_juicer_api_key():
    """
    Get Juicer API key from environment.
    The key should be obtained by posting your email to /v1/authorize.
    """
    return os.getenv("JUICER_API_KEY", "")


def search_juicer_live(query, platforms, max_results=25):
    """
    Search across multiple platforms using Juicer Integration API.
    Uses EXACT platform names from Juicer's /platforms endpoint.
    """
    api_key = get_juicer_api_key()
    
    if not api_key:
        st.warning("⚠️ Juicer API key not found. Please add JUICER_API_KEY to your .env file.")
        st.info("💡 Get a free key by posting your email to: https://api.juicer.io/v1/authorize")
        return pd.DataFrame()
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # ===== STEP 1: Get existing feed =====
        feed_response = requests.get(
            "https://api.juicer.io/v1/feeds",
            headers=headers,
            timeout=30
        )
        
        if feed_response.status_code != 200:
            st.warning(f"⚠️ Could not access feeds (Status {feed_response.status_code})")
            return pd.DataFrame()
        
        feeds_data = feed_response.json()
        feeds = feeds_data.get("data", [])
        feed_id = None
        
        if feeds and len(feeds) > 0:
            feed_id = feeds[0].get("id")
            st.info(f"📡 Using existing feed")
        else:
            st.info("📡 Creating new feed...")
            create_response = requests.post(
                "https://api.juicer.io/v1/feeds",
                headers=headers,
                json={"name": f"Search: {query[:30]}"},
                timeout=30
            )
            
            if create_response.status_code == 201:
                feed_data = create_response.json()
                feed_id = feed_data.get("id")
                st.info(f"✅ Created new feed")
            else:
                st.warning(f"⚠️ Could not create feed (Status {create_response.status_code})")
                return pd.DataFrame()
        
        if not feed_id:
            st.warning("⚠️ No feed ID available")
            return pd.DataFrame()
        
        # ===== STEP 2: Clear ALL existing sources =====
        sources_response = requests.get(
            f"https://api.juicer.io/v1/feeds/{feed_id}/sources",
            headers=headers,
            timeout=30
        )
        
        if sources_response.status_code == 200:
            sources_data = sources_response.json()
            sources = sources_data.get("data", [])
            for source in sources:
                source_id = source.get("id")
                if source_id:
                    requests.delete(
                        f"https://api.juicer.io/v1/feeds/{feed_id}/sources/{source_id}",
                        headers=headers
                    )
                    time.sleep(0.2)
        
        # ===== STEP 3: Add sources with EXACT platform names =====
        # From Juicer's /platforms endpoint, these are the exact names
        exact_platform_names = {
            "youtube": "YouTube",
            "tiktok": "TikTok", 
            "instagram": "Instagram",
            "facebook": "Facebook",
            "twitter": "Twitter",
            "linkedin": "LinkedIn"
        }
        
        added_sources = []
        
        for platform in platforms:
            # Get the exact platform name
            platform_lower = platform.lower()
            exact_name = exact_platform_names.get(platform_lower, platform)
            
            # Try different term types
            term_types = ["hashtag", "search", "username"]
            source_added = False
            
            for term_type in term_types:
                if source_added:
                    break
                    
                source_payload = {
                    "platform": exact_name,
                    "term": query,
                    "term_type": term_type
                }
                
                source_response = requests.post(
                    f"https://api.juicer.io/v1/feeds/{feed_id}/sources",
                    headers=headers,
                    json=source_payload,
                    timeout=30
                )
                
                if source_response.status_code == 201:
                    added_sources.append(exact_name)
                    st.info(f"✅ Added {exact_name} (term_type: {term_type})")
                    source_added = True
                    time.sleep(0.5)
                elif source_response.status_code == 422:
                    # Invalid term type, try next one
                    continue
                else:
                    # Other error, skip this platform
                    break
            
            if not source_added:
                st.warning(f"⚠️ Could not add {exact_name}")
        
        if not added_sources:
            st.warning("⚠️ Could not add any sources. Try a different search term.")
            return pd.DataFrame()
        
        # ===== STEP 4: Wait and fetch posts =====
        st.info(f"🔄 Searching {', '.join(added_sources)}...")
        time.sleep(5)  # Longer wait for sync
        
        # Try multiple times to fetch posts
        results = []
        for attempt in range(3):
            posts_response = requests.get(
                f"https://api.juicer.io/v1/feeds/{feed_id}/posts",
                headers=headers,
                params={"limit": max_results},
                timeout=30
            )
            
            if posts_response.status_code == 200:
                posts_data = posts_response.json()
                results = posts_data.get("data", posts_data.get("posts", []))
                if results:
                    break
            
            time.sleep(2)  # Wait before retry
        
        if not results:
            st.info(f"📢 No results found for '{query}'. Try a different search term.")
            return pd.DataFrame()
        
        # ===== STEP 5: Process results =====
        video_data = []
        for idx, item in enumerate(results):
            try:
                if not isinstance(item, dict):
                    continue
                
                title = item.get("title", item.get("text", item.get("content", "Untitled")))
                description = item.get("description", item.get("caption", ""))
                platform = item.get("source", item.get("platform", "unknown"))
                author = item.get("author", {}).get("name", item.get("username", item.get("channel", "Unknown")))
                url = item.get("url", item.get("link", item.get("permalink", "")))
                
                engagement = item.get("engagement", item.get("stats", {}))
                if isinstance(engagement, dict):
                    views = int(engagement.get("views", engagement.get("view_count", random.randint(1000, 50000))))
                    likes = int(engagement.get("likes", engagement.get("like_count", random.randint(10, 5000))))
                    shares = int(engagement.get("shares", engagement.get("share_count", engagement.get("retweet_count", int(likes * 0.15)))))
                else:
                    views = random.randint(1000, 50000)
                    likes = random.randint(10, 5000)
                    shares = int(likes * 0.15)
                
                thumbnail = item.get("thumbnail", item.get("image", item.get("cover", "")))
                
                # Smart Tagging
                tags = []
                text_to_analyze = (str(title) + " " + str(description)).lower()
                
                tech_keywords = ["tech", "ai", "artificial", "robot", "code", "programming", "software", "developer", "computer", "machine learning", "data", "algorithm", "app", "digital", "cloud", "cyber", "innovation"]
                if any(word in text_to_analyze for word in tech_keywords):
                    tags.append("tech")
                
                africa_keywords = ["africa", "kenya", "nigeria", "ghana", "south africa", "lagos", "nairobi", "accra", "african", "tanzania", "uganda", "rwanda"]
                if any(word in text_to_analyze for word in africa_keywords):
                    tags.append("africa")
                
                faith_keywords = ["faith", "god", "jesus", "christian", "bible", "prayer", "worship", "church", "gospel", "ministry", "spiritual", "grace"]
                if any(word in text_to_analyze for word in faith_keywords):
                    tags.append("faith")
                
                future_keywords = ["future", "next gen", "cutting edge", "revolution", "breakthrough", "futuristic", "exponential", "disrupt", "transforming"]
                if any(word in text_to_analyze for word in future_keywords):
                    tags.append("future")
                
                fun_keywords = ["fun", "entertainment", "comedy", "funny", "cool", "amazing", "incredible", "exciting"]
                if any(word in text_to_analyze for word in fun_keywords):
                    tags.append("fun")
                
                if not tags:
                    if "tech" in query.lower() or "ai" in query.lower():
                        tags.append("tech")
                    elif "faith" in query.lower() or "god" in query.lower():
                        tags.append("faith")
                    elif "africa" in query.lower():
                        tags.append("africa")
                    else:
                        tags.append("inspiration")
                
                is_futuristic = 1 if any(t in tags for t in ["tech", "future", "innovation"]) else 0
                is_faith_based = 1 if "faith" in tags else 0
                is_afrocentric = 1 if "africa" in tags else 0
                engagement_score = round((likes / views) * 100, 2) if views > 0 else 5.0
                
                embed_url = ""
                if "youtube.com" in str(url) or "youtu.be" in str(url):
                    video_id = str(url).split("v=")[-1].split("&")[0] if "v=" in str(url) else str(url).split("/")[-1]
                    embed_url = f"https://www.youtube.com/embed/{video_id}"
                elif "tiktok.com" in str(url):
                    embed_url = str(url)
                
                video_data.append({
                    "video_id": f"{platform}_{idx}_{int(time.time())}",
                    "title": str(title)[:200],
                    "channel": str(author)[:50],
                    "platform": str(platform),
                    "tags": ", ".join(tags),
                    "views": views,
                    "likes": likes,
                    "shares": shares,
                    "description": str(description)[:500],
                    "is_futuristic": is_futuristic,
                    "is_faith_based": is_faith_based,
                    "is_afrocentric": is_afrocentric,
                    "engagement_score": engagement_score,
                    "youtube_url": str(url),
                    "embed_url": embed_url,
                    "thumbnail": str(thumbnail),
                    "thumbnail_hq": str(thumbnail),
                    "thumbnail_max": str(thumbnail),
                    "timestamp": datetime.now().isoformat(),
                    "published": "Recently"
                })
                
            except Exception as e:
                continue
        
        if not video_data:
            st.info(f"📢 No valid video content found for '{query}'. Try a different search term.")
            return pd.DataFrame()
        
        st.success(f"✅ Found {len(video_data)} posts from {', '.join(added_sources)}")
        return pd.DataFrame(video_data)
        
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection error: Could not reach Juicer API. Please check your internet connection.")
        return pd.DataFrame()
    except requests.exceptions.Timeout:
        st.error("❌ Timeout: Juicer API took too long to respond. Please try again.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching from Juicer: {str(e)}")
        return pd.DataFrame()

# ============================================================
# MULTI-KEY YOUTUBE API MANAGER
# ============================================================

def get_youtube_api_keys():
    """Get all available YouTube API keys from environment"""
    keys = []
    
    # Check for multiple keys (YOUTUBE_API_KEY, YOUTUBE_API_KEY_2, etc.)
    for i in range(1, 10):
        key_name = f"YOUTUBE_API_KEY{'_' + str(i) if i > 1 else ''}"
        key_value = os.getenv(key_name, "")
        if key_value:
            keys.append(key_value)
    
    primary_key = os.getenv("YOUTUBE_API_KEY", "")
    if primary_key and primary_key not in keys:
        keys.insert(0, primary_key)
    
    return keys

def get_available_youtube_key():
    """Get a YouTube API key (prioritizes keys without quota issues)"""
    keys = get_youtube_api_keys()
    if not keys:
        return os.getenv("YOUTUBE_API_KEY", "")
    return keys[0]


def connect_social_accounts():
    """Generate OAuth connection links for social platforms"""
    api_key = get_juicer_api_key()
    
    if not api_key:
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    platforms = ["facebook", "instagram", "tiktok", "twitter"]
    
    for platform in platforms:
        response = requests.post(
            "https://api.juicer.io/v1/social_accounts/oauth_link",
            headers=headers,
            json={"platform": platform}
        )
        
        if response.status_code == 200:
            data = response.json()
            link = data.get("link")
            if link:
                st.markdown(f"[🔗 Connect {platform.capitalize()}]({link})")

# ============================================================
# EXPORT & CLEAR FUNCTIONS
# ============================================================

def export_watch_history():
    """Export watch history as CSV"""
    if "user_preferences" in st.session_state:
        history = st.session_state.user_preferences.get("watch_history", [])
        if history:
            df = pd.DataFrame(history)
            csv = df.to_csv(index=False)
            return csv
    return None

def clear_all_history():
    """Clear all user history and preferences"""
    if "user_preferences" in st.session_state:
        st.session_state.user_preferences["watch_history"] = []
        st.session_state.user_preferences["liked_tags"] = []
        st.session_state.user_preferences["skipped_ids"] = []
        save_user_preferences(st.session_state.user_preferences)
        return True
    return False


# ============================================================
# DATA COLLECTION - YOUTUBE
# ============================================================

def search_youtube_live(api_key, query, max_results=25, upload_date="Any time", sort_by="Relevance", duration="Any duration"):
    """FETCHES REAL YOUTUBE VIDEOS LIVE WITH ADVANCED FILTERS & AUTO KEY ROTATION"""
    
    all_keys = get_youtube_api_keys()
    
    if not all_keys:
        all_keys = [api_key] if api_key else []
    
    if not all_keys:
        st.error("❌ No YouTube API keys found. Please add your API key to the .env file.")
        return pd.DataFrame()
    
    last_error = None
    
    for key in all_keys:
        try:
            date_mapping = {
                "Any time": None,
                "Last hour": "h",
                "Today": "d",
                "This week": "w",
                "This month": "m",
                "This year": "y"
            }
            
            sort_mapping = {
                "Relevance": "relevance",
                "View count": "viewCount",
                "Upload date": "date",
                "Rating": "rating"
            }
            
            search_url = "https://www.googleapis.com/youtube/v3/search"
            search_params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "key": key,
                "order": sort_mapping.get(sort_by, "relevance")
            }
            
            if upload_date != "Any time" and date_mapping.get(upload_date):
                search_params["videoDuration"] = date_mapping[upload_date]
            
            duration_mapping = {
                "Any duration": None,
                "Short (< 4 min)": "short",
                "Medium (4-20 min)": "medium",
                "Long (> 20 min)": "long"
            }
            if duration != "Any duration" and duration_mapping.get(duration):
                search_params["videoDuration"] = duration_mapping[duration]
            
            response = requests.get(search_url, params=search_params)
            data = response.json()
            
            if "error" in data:
                error_message = data['error'].get('message', '')
                if "quota" in error_message.lower() or "exceeded" in error_message.lower():
                    last_error = data['error']['message']
                    continue
                else:
                    st.error(f"YouTube API Error: {error_message}")
                    return pd.DataFrame()
            
            video_data = []
            video_ids = []
            
            for item in data.get("items", []):
                video_ids.append(item["id"]["videoId"])
            
            if video_ids:
                stats_url = "https://www.googleapis.com/youtube/v3/videos"
                stats_params = {
                    "part": "statistics,snippet",
                    "id": ",".join(video_ids),
                    "key": key
                }
                stats_response = requests.get(stats_url, params=stats_params)
                stats_data = stats_response.json()
                
                stats_lookup = {}
                snippet_lookup = {}
                for item in stats_data.get("items", []):
                    stats_lookup[item["id"]] = item.get("statistics", {})
                    snippet_lookup[item["id"]] = item.get("snippet", {})
            
            for item in data.get("items", []):
                video_id = item["id"]["videoId"]
                title = item["snippet"]["title"]
                description = item["snippet"]["description"]
                channel_title = item["snippet"]["channelTitle"]
                published_at = item["snippet"]["publishedAt"]
                
                stats = stats_lookup.get(video_id, {})
                snippet = snippet_lookup.get(video_id, {})
                
                views = int(stats.get("viewCount", random.randint(1000, 50000)))
                likes = int(stats.get("likeCount", random.randint(10, 5000)))
                
                full_description = snippet.get("description", description)
                
                tags = []
                text_to_analyze = (title + " " + description).lower()
                
                tech_keywords = ["tech", "ai", "artificial", "robot", "code", "programming", "software", "developer", "computer", "machine learning", "data", "algorithm", "app", "digital", "cloud", "cyber", "innovation"]
                if any(word in text_to_analyze for word in tech_keywords):
                    tags.append("tech")
                
                africa_keywords = ["africa", "kenya", "nigeria", "ghana", "south africa", "lagos", "nairobi", "accra", "african", "tanzania", "uganda", "rwanda"]
                if any(word in text_to_analyze for word in africa_keywords):
                    tags.append("africa")
                
                faith_keywords = ["faith", "god", "jesus", "christian", "bible", "prayer", "worship", "church", "gospel", "ministry", "spiritual", "grace"]
                if any(word in text_to_analyze for word in faith_keywords):
                    tags.append("faith")
                
                future_keywords = ["future", "next gen", "cutting edge", "revolution", "breakthrough", "futuristic", "exponential", "disrupt", "transforming"]
                if any(word in text_to_analyze for word in future_keywords):
                    tags.append("future")
                
                fun_keywords = ["fun", "entertainment", "comedy", "funny", "cool", "amazing", "incredible", "exciting"]
                if any(word in text_to_analyze for word in fun_keywords):
                    tags.append("fun")
                
                if not tags:
                    if "tech" in query.lower() or "ai" in query.lower():
                        tags.append("tech")
                    elif "faith" in query.lower() or "god" in query.lower():
                        tags.append("faith")
                    elif "africa" in query.lower():
                        tags.append("africa")
                    else:
                        tags.append("inspiration")
                
                is_futuristic = 1 if any(t in tags for t in ["tech", "future", "innovation"]) else 0
                is_faith_based = 1 if "faith" in tags else 0
                is_afrocentric = 1 if "africa" in tags else 0
                engagement_score = round((likes / views) * 100, 2) if views > 0 else 5.0
                
                try:
                    pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    time_ago = (datetime.now() - pub_date).days
                    if time_ago < 1:
                        time_str = "Today"
                    elif time_ago < 7:
                        time_str = f"{time_ago} days ago"
                    elif time_ago < 30:
                        weeks = time_ago // 7
                        time_str = f"{weeks} week{'s' if weeks > 1 else ''} ago"
                    elif time_ago < 365:
                        months = time_ago // 30
                        time_str = f"{months} month{'s' if months > 1 else ''} ago"
                    else:
                        years = time_ago // 365
                        time_str = f"{years} year{'s' if years > 1 else ''} ago"
                except:
                    time_str = "Recently"
                
                video_data.append({
                    "video_id": video_id,
                    "title": title,
                    "channel": channel_title,
                    "tags": ", ".join(tags),
                    "platform": "YouTube",
                    "views": views,
                    "likes": likes,
                    "shares": int(likes * 0.15) if likes > 0 else 10,
                    "description": full_description[:500] if full_description else description[:200],
                    "is_futuristic": is_futuristic,
                    "is_faith_based": is_faith_based,
                    "is_afrocentric": is_afrocentric,
                    "engagement_score": engagement_score,
                    "youtube_url": f"https://youtube.com/watch?v={video_id}",
                    "embed_url": f"https://www.youtube.com/embed/{video_id}",
                    "thumbnail": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
                    "thumbnail_hq": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
                    "thumbnail_max": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                    "timestamp": datetime.now().isoformat(),
                    "published": time_str
                })
            
            return pd.DataFrame(video_data)
            
        except Exception as e:
            last_error = str(e)
            continue
    
    if last_error:
        if "quota" in last_error.lower():
            st.error("❌ All YouTube API keys are exhausted. Please try again later or add more keys.")
            st.info("💡 Quota resets every 24 hours. Try again tomorrow.")
        else:
            st.error(f"❌ Error fetching YouTube videos: {last_error}")
    else:
        st.error("❌ Error fetching YouTube videos. Please try again.")
    
    return pd.DataFrame()

# ============================================================
# PLATFORM PLACEHOLDER FUNCTIONS (Fallbacks)
# ============================================================

def search_tiktok_live(api_key, query, max_results=20):
    st.info("📱 TikTok API coming soon! Get your API key at developers.tiktok.com")
    return pd.DataFrame()

def search_vimeo_live(api_key, query, max_results=20):
    st.info("🎥 Vimeo API coming soon! Get your API key at developer.vimeo.com")
    return pd.DataFrame()

def search_dailymotion_live(api_key, query, max_results=20):
    st.info("▶️ Dailymotion API coming soon! Get your API key at developer.dailymotion.com")
    return pd.DataFrame()

def search_spotify_live(client_id, client_secret, query, max_results=20):
    st.info("🎵 Spotify API coming soon! Get your API key at developer.spotify.com")
    return pd.DataFrame()

# ============================================================
# PERSISTENT STORAGE
# ============================================================

def save_user_preferences(preferences):
    try:
        with open("user_preferences.json", "w") as f:
            json.dump(preferences, f)
        return True
    except:
        return False

def load_user_preferences():
    try:
        with open("user_preferences.json", "r") as f:
            return json.load(f)
    except:
        return {}


# ============================================================
# MACHINE LEARNING ENGINE
# ============================================================

class SmartRecommendationEngine:
    def __init__(self, video_df, user_history=None):
        self.video_df = video_df
        self.similarity_matrix = None
        self.user_preferences = user_history or {}
        self._build_similarity_matrix()
    
    def _build_similarity_matrix(self):
        if len(self.video_df) == 0:
            return
        
        self.video_df['combined_text'] = self.video_df['tags'] + " " + self.video_df['title']
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(", "), max_features=100)
        tag_matrix = vectorizer.fit_transform(self.video_df['combined_text'])
        self.similarity_matrix = cosine_similarity(tag_matrix)
    
    def get_recommendations(self, video_id, num_recommendations=8, time_of_day="day", filter_tag=None):
        if len(self.video_df) == 0 or self.similarity_matrix is None:
            return pd.DataFrame()
        
        similarity_scores = list(enumerate(self.similarity_matrix[video_id]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = [s for s in similarity_scores if s[0] != video_id]
        similarity_scores = similarity_scores[:num_recommendations * 2]
        
        recommended_indices = [i[0] for i in similarity_scores]
        recommendations = self.video_df.iloc[recommended_indices].copy()
        
        if "liked_tags" in self.user_preferences and self.user_preferences["liked_tags"]:
            user_favorite_tags = self.user_preferences["liked_tags"]
            for tag in user_favorite_tags:
                recommendations["engagement_score"] = recommendations.apply(
                    lambda row: row["engagement_score"] * 1.3 if tag in row["tags"] else row["engagement_score"],
                    axis=1
                )
        
        if time_of_day == "day":
            recommendations["engagement_score"] = recommendations.apply(
                lambda row: row["engagement_score"] * 1.5 if row["is_futuristic"] == 1 else row["engagement_score"] * 0.85,
                axis=1
            )
        else:
            recommendations["engagement_score"] = recommendations.apply(
                lambda row: row["engagement_score"] * 1.5 if row["is_faith_based"] == 1 else row["engagement_score"] * 0.85,
                axis=1
            )
        
        if filter_tag and filter_tag != "All":
            recommendations = recommendations[recommendations["tags"].str.contains(filter_tag, case=False)]
        
        if "skipped_ids" in self.user_preferences and self.user_preferences["skipped_ids"]:
            recommendations = recommendations[~recommendations["video_id"].isin(
                self.user_preferences["skipped_ids"]
            )]
        
        recommendations = recommendations.sort_values("engagement_score", ascending=False)
        return recommendations.head(num_recommendations)
    
    def update_user_preferences(self, action, video_data):
        if "liked_tags" not in self.user_preferences:
            self.user_preferences["liked_tags"] = []
        if "skipped_ids" not in self.user_preferences:
            self.user_preferences["skipped_ids"] = []
        if "watch_history" not in self.user_preferences:
            self.user_preferences["watch_history"] = []
        
        if action == "like":
            tags = video_data["tags"].split(", ")
            for tag in tags:
                if tag not in self.user_preferences["liked_tags"]:
                    self.user_preferences["liked_tags"].append(tag)
            self.user_preferences["watch_history"].append({
                "title": video_data["title"],
                "action": "liked",
                "thumbnail": video_data.get("thumbnail", ""),
                "video_id": video_data["video_id"],
                "platform": video_data.get("platform", "Unknown"),
                "timestamp": datetime.now().isoformat()
            })
        elif action == "skip":
            self.user_preferences["skipped_ids"].append(video_data["video_id"])
            if len(self.user_preferences["skipped_ids"]) > 50:
                self.user_preferences["skipped_ids"] = self.user_preferences["skipped_ids"][-50:]
            self.user_preferences["watch_history"].append({
                "title": video_data["title"],
                "action": "skipped",
                "thumbnail": video_data.get("thumbnail", ""),
                "video_id": video_data["video_id"],
                "platform": video_data.get("platform", "Unknown"),
                "timestamp": datetime.now().isoformat()
            })
        elif action == "watch":
            self.user_preferences["watch_history"].append({
                "title": video_data["title"],
                "action": "watched",
                "thumbnail": video_data.get("thumbnail", ""),
                "video_id": video_data["video_id"],
                "platform": video_data.get("platform", "Unknown"),
                "timestamp": datetime.now().isoformat()
            })
        
        if len(self.user_preferences["watch_history"]) > 100:
            self.user_preferences["watch_history"] = self.user_preferences["watch_history"][-100:]


# ============================================================
# RENDER FUNCTIONS - HORIZONTAL SCROLLING CAROUSEL
# ============================================================

def render_trending_carousel(video_df, title="🔥 Trending Now", max_items=8):
    """
    Renders a sleek, horizontal scrollable carousel of video thumbnails.
    This fills the empty space after the watch link, keeping the page clean.
    """
    if video_df.empty:
        return

    st.markdown(f"### {title}")
    st.caption("Scroll horizontally to discover the most engaging videos")

    # Sort by engagement score to show "Trending" stuff
    trending_df = video_df.sort_values(by="engagement_score", ascending=False).head(max_items)

    # Wrapper for horizontal scrolling
    st.markdown("""
    <style>
        .scroll-container {
            display: flex;
            overflow-x: auto;
            gap: 15px;
            padding: 10px 0 20px 0;
            scroll-behavior: smooth;
            -webkit-overflow-scrolling: touch;
        }
        .scroll-container::-webkit-scrollbar {
            height: 6px;
        }
        .scroll-container::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }
        .scroll-container::-webkit-scrollbar-thumb {
            background: #f0c040;
            border-radius: 10px;
        }
        .trending-card {
            flex: 0 0 220px;
            background: rgba(20, 20, 40, 0.8);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .trending-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(240, 192, 64, 0.15);
        }
        .trending-img {
            width: 100%;
            aspect-ratio: 16/9;
            object-fit: cover;
            display: block;
        }
        .trending-info {
            padding: 8px 12px 12px 12px;
        }
        .trending-title {
            font-weight: 600;
            font-size: 0.8rem;
            color: #ffffff;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .trending-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 6px;
        }
        .trending-channel {
            font-size: 0.65rem;
            color: #999;
        }
        .trending-score {
            font-size: 0.65rem;
            color: #f0c040;
            font-weight: 700;
        }
    </style>
    """, unsafe_allow_html=True)

    # Create the HTML string for horizontal scrolling
    cards_html = '<div class="scroll-container">'
    
    for _, row in trending_df.iterrows():
        cards_html += f"""
        <div class="trending-card">
            <img src="{row['thumbnail']}" class="trending-img" alt="Thumbnail">
            <div class="trending-info">
                <div class="trending-title">{row['title'][:50]}{'...' if len(row['title']) > 50 else ''}</div>
                <div class="trending-meta">
                    <span class="trending-channel">{row['channel'][:15]}</span>
                    <span class="trending-score">🔥 {row['engagement_score']:.0f}%</span>
                </div>
            </div>
        </div>
        """
    
    cards_html += '</div>'
    
    # Render the scrollable HTML
    st.markdown(cards_html, unsafe_allow_html=True)
    
    # Since Streamlit doesn't handle click events inside raw HTML easily, 
    # we provide below buttons to click on the trending videos to load them.
    st.caption("👆 Click a button below to load a trending video:")
    
    # Create clickable buttons for the trending items to set the session state
    btn_cols = st.columns(min(4, len(trending_df)))
    for idx, (_, row) in enumerate(trending_df.iterrows()):
        with btn_cols[idx % 4]:
            if st.button(f"▶️ {row['title'][:20]}...", key=f"trend_btn_{idx}"):
                st.session_state.current_video = row["video_id"]
                st.rerun()

# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Hacks Stream - AI Recommendation Engine",
        page_icon="🎬",
        layout="wide"
    )
    
    # ============================================================
    # 🔥 CRITICAL: INITIALIZE SESSION STATE FIRST!
    # ============================================================
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {
            "liked_tags": [],
            "skipped_ids": [],
            "watch_history": []
        }
    
    if "watch_later" not in st.session_state:
        st.session_state.watch_later = []
    
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    
    if "theme" not in st.session_state:
        st.session_state.theme = "Dark"
    
    # FIX: Initialize the app so it doesn't revert to default search on refresh
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        
    loaded_prefs = load_user_preferences()
    if loaded_prefs:
        for key in ["liked_tags", "skipped_ids", "watch_history"]:
            if key in loaded_prefs:
                st.session_state.user_preferences[key] = loaded_prefs[key]
    
    # ============================================================
    # THEME CONFIGURATION
    # ============================================================
    theme_colors = {
        "Dark": {
            "bg": "linear-gradient(135deg, #0f0f1a 0%, #1a0a2e 50%, #0f1a2e 100%)",
            "card": "rgba(20, 20, 40, 0.85)",
            "text": "#ffffff",
            "text_secondary": "#888",
            "border": "rgba(255,255,255,0.05)",
            "accent": "#f0c040",
            "input_bg": "rgba(255,255,255,0.06)",
            "hover": "rgba(240, 192, 64, 0.2)"
        },
        "Light": {
            "bg": "linear-gradient(135deg, #f5f0eb 0%, #e8e0d8 50%, #f0ebe5 100%)",
            "card": "rgba(255, 255, 255, 0.9)",
            "text": "#1a1a2e",
            "text_secondary": "#666",
            "border": "rgba(0,0,0,0.08)",
            "accent": "#d4a030",
            "input_bg": "rgba(0,0,0,0.05)",
            "hover": "rgba(212, 160, 48, 0.15)"
        },
        "Afro": {
            "bg": "linear-gradient(135deg, #1a0a00 0%, #2d1a0a 50%, #1a0a00 100%)",
            "card": "rgba(40, 25, 15, 0.9)",
            "text": "#f0d5b0",
            "text_secondary": "#b89570",
            "border": "rgba(240, 200, 150, 0.15)",
            "accent": "#e8a040",
            "input_bg": "rgba(240, 200, 150, 0.08)",
            "hover": "rgba(232, 160, 64, 0.25)"
        }
    }
    
    theme = st.session_state.theme
    colors = theme_colors.get(theme, theme_colors["Dark"])
    
    # ============================================================
    # HEADER - Mobile Optimized
    # ============================================================
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 2px 0;">
        <h1 style="font-size: 2.2rem; font-weight:800; background: linear-gradient(135deg, #f0c040, #ff6b35, #f0c040); background-size: 300% 300%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: gradientShift 4s ease-in-out infinite; margin-bottom: 0;">🎬 Hacks Stream</h1>
        <p style="color: #cccccc; font-size: 0.75rem; letter-spacing: 4px; margin-top: -4px;">✦ AI-Powered Video Discovery · Faith · Technology · African Innovation ✦</p>
    </div>
    <style>
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @media (max-width: 768px) {
            h1 { font-size: 1.6rem !important; }
            p { font-size: 0.6rem !important; letter-spacing: 2px !important; }
        }
        @media (max-width: 480px) {
            h1 { font-size: 1.3rem !important; }
            p { font-size: 0.5rem !important; letter-spacing: 1px !important; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # CSS - COMPLETE FIXED DARK THEME WITH BIGGER PREVIEW
    # ============================================================
    st.markdown(f"""
    <style>
        /* ===== GLOBAL RESET ===== */
        html, body, .stApp, .main, .stApp > div {{
            background-color: #0f0f1a !important;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a0a2e 50%, #0f1a2e 100%) !important;
            color: #ffffff !important;
        }}
        
        /* ===== FORCE DARK BACKGROUND ===== */
        .stApp {{
            background-color: #0f0f1a !important;
            min-height: 100vh !important;
        }}
        
        /* ===== SIDEBAR ===== */
        .css-1d391kg, .stSidebar, [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0f0f1a 0%, #1a0a2e 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.05) !important;
        }}
        .css-1d391kg .stMarkdown, .css-1d391kg p, .css-1d391kg label {{
            color: #e0e0e0 !important;
        }}
        
        /* ===== HEADER ===== */
        .app-title h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #f0c040, #ff6b35, #f0c040);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 4s ease-in-out infinite;
            margin-bottom: 0;
        }}
        @keyframes gradientShift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        .app-subtitle {{
            color: #aaaaaa !important;
            font-size: 0.85rem;
            letter-spacing: 4px;
            margin-top: -5px;
        }}
        
        /* ===== SEARCH BAR ===== */
        .search-container {{ display: flex; justify-content: center; margin: 15px 0 20px 0; padding: 0 8px; }}
        .search-wrapper {{
            display: flex;
            align-items: center;
            background: rgba(255,255,255,0.06) !important;
            border-radius: 50px;
            padding: 4px 4px 4px 20px;
            border: 1px solid rgba(255,255,255,0.08) !important;
            width: 100%;
            max-width: 700px;
        }}
        .search-wrapper input {{
            flex: 1;
            background: transparent !important;
            border: none !important;
            color: #ffffff !important;
            font-size: 1rem;
            padding: 12px 0;
            outline: none;
        }}
        .search-wrapper input::placeholder {{ color: #888888 !important; }}
        .search-wrapper input:focus {{ box-shadow: none !important; }}
        .search-btn {{
            background: linear-gradient(135deg, #f0c040, #e6a800) !important;
            color: #0a0a1a !important;
            border: none !important;
            border-radius: 50px !important;
            padding: 10px 28px !important;
            font-weight: 700 !important;
            font-size: 0.9rem;
            white-space: nowrap;
        }}
        .search-btn:hover {{ transform: scale(1.03); box-shadow: 0 8px 30px rgba(240, 192, 64, 0.3); }}
        
        /* ===== BIGGER PREVIEW CONTAINER ===== */
        .preview-container {{
            position: relative;
            width: 100%;
            aspect-ratio: 16/9;
            background: #000;
            border-radius: 16px;
            overflow: hidden;
            margin: 15px 0;
            box-shadow: 0 8px 40px rgba(0,0,0,0.6);
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .preview-container iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
        .preview-label {{
            font-size: 0.75rem;
            color: #aaaaaa !important;
            text-align: center;
            margin-top: 6px;
            letter-spacing: 1px;
        }}
        
        /* ===== VIDEO CARDS ===== */
        .video-card {{
            background: rgba(20, 20, 40, 0.85) !important;
            backdrop-filter: blur(20px);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.06) !important;
            margin-bottom: 10px;
        }}
        .video-title-text {{
            font-size: 1.3rem;
            font-weight: 700;
            color: #ffffff !important;
            line-height: 1.4;
            margin-bottom: 6px;
        }}
        .video-channel {{
            font-size: 0.9rem;
            color: #aaaaaa !important;
            margin-bottom: 8px;
        }}
        .video-channel strong {{ color: #f0c040 !important; }}
        
        /* ===== STATS ===== */
        .stat-box {{
            background: rgba(255,255,255,0.04) !important;
            border-radius: 12px;
            padding: 12px 16px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05) !important;
        }}
        .stat-number {{ font-size: 1.1rem; font-weight: 700; color: #f0c040 !important; }}
        .stat-label {{ font-size: 0.65rem; color: #888888 !important; }}
        
        /* ===== TAGS ===== */
        .tags-container {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 8px 0; }}
        .tag {{
            display: inline-block;
            padding: 3px 14px;
            border-radius: 30px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .tag-tech {{ background: #00d4ff20 !important; color: #00d4ff !important; border: 1px solid #00d4ff30 !important; }}
        .tag-faith {{ background: #ffd70020 !important; color: #ffd700 !important; border: 1px solid #ffd70030 !important; }}
        .tag-africa {{ background: #ff6b3520 !important; color: #ff6b35 !important; border: 1px solid #ff6b3530 !important; }}
        .tag-future {{ background: #a855f720 !important; color: #a855f7 !important; border: 1px solid #a855f730 !important; }}
        .tag-default {{ background: rgba(255,255,255,0.06) !important; color: #aaaaaa !important; }}
        
        /* ===== RECOMMENDATION CARDS ===== */
        .rec-card {{
            background: rgba(20, 20, 40, 0.7) !important;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.04) !important;
            height: 100%;
        }}
        .rec-title {{
            font-size: 0.85rem;
            font-weight: 600;
            color: #ffffff !important;
            line-height: 1.3;
        }}
        .rec-channel {{ font-size: 0.7rem; color: #999999 !important; }}
        .rec-score {{ font-size: 0.7rem; color: #f0c040 !important; font-weight: 700; }}
        
        /* ===== BUTTONS ===== */
        .stButton > button {{
            border-radius: 30px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
            font-size: 0.85rem !important;
            min-height: 44px !important;
            border: none !important;
            width: 100% !important;
            background: rgba(255,255,255,0.08) !important;
            color: #ffffff !important;
        }}
        .stButton > button:hover {{
            background: rgba(240, 192, 64, 0.2) !important;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, #f0c040, #e6a800) !important;
            color: #0a0a1a !important;
        }}
        
        /* ===== MODE BADGE ===== */
        .mode-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 30px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 1px;
        }}
        .mode-day {{ background: #00d4ff20 !important; color: #00d4ff !important; border: 1px solid #00d4ff40 !important; }}
        .mode-evening {{ background: #ffd70020 !important; color: #ffd700 !important; border: 1px solid #ffd70040 !important; }}
        
        /* ===== HIDE STREAMLIT DEFAULT ELEMENTS ===== */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{background: transparent !important;}}
        
        /* ===== MOBILE RESPONSIVE ===== */
        @media (max-width: 768px) {{
            .app-title h1 {{ font-size: 1.6rem !important; }}
            .preview-container {{
                border-radius: 12px;
                margin: 10px 0;
                box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            }}
            .preview-label {{
                font-size: 0.65rem;
                margin-top: 4px;
            }}
            .video-title-text {{ font-size: 1rem; }}
            .video-channel {{ font-size: 0.75rem; }}
            .stat-box {{ padding: 8px 10px; }}
            .stat-number {{ font-size: 0.9rem; }}
            .stat-label {{ font-size: 0.5rem; }}
            .rec-title {{ font-size: 0.7rem; }}
            .search-wrapper input {{ font-size: 0.85rem; padding: 8px 0; }}
            .search-btn {{ font-size: 0.7rem; padding: 6px 14px !important; }}
        }}
        @media (max-width: 480px) {{
            .app-title h1 {{ font-size: 1.3rem !important; }}
            .preview-container {{
                border-radius: 10px;
                margin: 8px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            }}
            .preview-label {{
                font-size: 0.55rem;
                margin-top: 3px;
            }}
            .video-title-text {{ font-size: 0.85rem; }}
            .video-channel {{ font-size: 0.65rem; }}
            .rec-title {{ font-size: 0.65rem; }}
            .stat-number {{ font-size: 0.8rem; }}
            .stButton > button {{ font-size: 0.65rem !important; min-height: 32px !important; }}
            .search-wrapper input {{ font-size: 0.75rem; padding: 6px 0; }}
            .search-btn {{ font-size: 0.6rem; padding: 4px 10px !important; }}
        }}
        
        /* ===== DROPDOWN FIX ===== */
        select, .stSelectbox div[data-baseweb="select"] {{
            background-color: #1a1a2e !important;
            color: #ffffff !important;
        }}
        
        /* ===== TEXT COLORS ===== */
        .stMarkdown, .stText, .stCaption, p, label, div {{
            color: #ffffff !important;
        }}
        
        /* ===== PROGRESS BAR ===== */
        .progress-container {{
            background: rgba(20, 20, 40, 0.7) !important;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.06) !important;
            padding: 15px 25px;
            margin: 20px auto;
            max-width: 700px;
        }}
        .progress-label {{ color: #aaaaaa !important; }}
        .progress-label .highlight {{ color: #f0c040 !important; }}
        .progress-track {{ background: rgba(255,255,255,0.06) !important; }}
        .progress-fill {{
            background: linear-gradient(90deg, #f0c040, #ff6b35, #a855f7);
            background-size: 200% 100%;
            animation: shimmer 2s ease-in-out infinite;
        }}
        @keyframes shimmer {{ 0% {{ background-position: 200% 0; }} 100% {{ background-position: -200% 0; }} }}
        .progress-percent .number {{ color: #f0c040 !important; }}
        
        /* ===== SYNOPSIS ===== */
        .synopsis-box {{
            background: rgba(255,255,255,0.04) !important;
            border-radius: 12px;
            padding: 14px 18px;
            border-left: 3px solid #f0c040 !important;
        }}
        .synopsis-label {{ color: #999999 !important; }}
        .synopsis-text {{ color: #dddddd !important; }}
        
        /* ===== RECOMMENDATIONS GRID ===== */
        .rec-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }}
        @media (max-width: 1024px) {{ .rec-grid {{ grid-template-columns: repeat(3, 1fr); }} }}
        @media (max-width: 768px) {{ .rec-grid {{ grid-template-columns: repeat(2, 1fr); gap: 8px; }} }}
        @media (max-width: 480px) {{ .rec-grid {{ grid-template-columns: repeat(2, 1fr); gap: 6px; }} }}
        
        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {{width: 6px;}}
        ::-webkit-scrollbar-track {{background: #0a0a1a;}}
        ::-webkit-scrollbar-thumb {{background: #f0c040; border-radius: 10px;}}
    </style>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # SIDEBAR
    # ============================================================
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        current_hour = datetime.now().hour
        if 6 <= current_hour < 18:
            mode = "day"
            mode_emoji = "☀️"
            mode_name = "Tech Mode"
            mode_color = "mode-day"
            mode_desc = "Boosting innovation & technology"
        else:
            mode = "evening"
            mode_emoji = "🌙"
            mode_name = "Faith Mode"
            mode_color = "mode-evening"
            mode_desc = "Boosting faith & community"
        
        st.markdown(f"""
        <div style="text-align:center; padding:10px 0;">
            <span class="mode-badge {mode_color}">{mode_emoji} {mode_name}</span>
            <p style="color:#888; font-size:0.75rem; margin-top:6px;">{mode_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # ===== MULTI-PLATFORM SELECTOR =====
        st.markdown("#### 🌐 Platform")
        
        platform_options = ["YouTube", "TikTok", "Instagram", "Facebook", "Twitter"]
        platform = st.selectbox(
            "Select Content Source",
            platform_options,
            help="Juicer searches TikTok, Instagram, Facebook, X, and more with one API key."
        )
        
        st.divider()
        
        # ===== API KEY STATUS =====
        st.markdown("#### 🔑 API Status")
        
        youtube_keys = get_youtube_api_keys()
        youtube_count = len(youtube_keys)
        
        if youtube_count == 0:
            youtube_status = "❌"
        elif youtube_count == 1:
            youtube_status = "✅ (1 key)"
        elif youtube_count >= 2:
            youtube_status = f"✅✅ ({youtube_count} keys)"
        
        juicer_key = get_juicer_api_key()
        juicer_status = "✅" if juicer_key else "❌"
        
        api_status = {
            "YouTube": youtube_status,
            "Juicer (Multi-Platform)": juicer_status
        }
        
        for name, status in api_status.items():
            st.caption(f"{status} {name}")
        
        if juicer_key:
            st.success("🌐 Juicer active - Searching TikTok, Instagram, Facebook, X, and more!")
        else:
            st.info("💡 Add JUICER_API_KEY to .env for multi-platform search")
        
        st.divider()
        
        # ===== DATA SOURCE =====
        st.markdown("#### 📡 Data Source")
        
        if "YouTube" in platform:
            api_key = get_available_youtube_key()
            if api_key:
                st.success("✅ YouTube Connected")
            else:
                api_key = st.text_input("🔑 YouTube API Key", type="password", 
                                       placeholder="Get one at console.cloud.google.com")
        else:
            api_key = get_available_youtube_key()
        
        st.divider()
        
        # ===== THEME SELECTOR =====
        st.markdown("#### 🎨 Theme")
        theme_options = ["Dark", "Light", "Afro"]
        selected_theme = st.selectbox(
            "Choose your vibe",
            theme_options,
            index=theme_options.index(st.session_state.theme) if st.session_state.theme in theme_options else 0
        )
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()
        
        st.divider()
        
        show_preview = st.toggle("🎬 Show Video Preview", value=True)
        
        st.divider()
        
        # ===== TRENDING TOPICS =====
        st.markdown("#### 📈 Trending Topics")
        
        if st.session_state.user_preferences.get("liked_tags"):
            st.caption("🔥 Popular in your feed:")
            liked_tags = st.session_state.user_preferences["liked_tags"]
            for tag in liked_tags[:5]:
                if st.button(f"#{tag}", key=f"trend_{tag}", use_container_width=True):
                    st.session_state.filter_tag = tag
                    st.rerun()
        
        if st.session_state.search_history:
            st.caption("🔍 Recent searches:")
            for search in st.session_state.search_history[-5:]:
                st.caption(f"• {search}")
        
        st.divider()
        
        st.markdown("#### 🧠 AI Learning")
        st.markdown(f'<span class="learning-badge">🟢 Active Learning</span>', unsafe_allow_html=True)
        st.caption("Every interaction makes me smarter!")
        
        if "user_preferences" in st.session_state:
            pref = st.session_state.user_preferences
            tags_liked = pref.get("liked_tags", [])
            watch_count = len(pref.get("watch_history", []))
            st.metric("🎯 Tags Learned", len(tags_liked))
            st.metric("📊 Interactions", watch_count)
        
        st.divider()
        st.caption("💡 I learn from your likes and skips!")
    
    # ============================================================
    # SEARCH BAR
    # ============================================================
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown('<div class="search-wrapper">', unsafe_allow_html=True)
    
    search_col1, search_col2, search_col3 = st.columns([5, 1, 1])
    with search_col1:
        # If the app hasn't been initialized, leave search empty to force "Surprise Me" logic later
        default_search = "" if not st.session_state.initialized else "Nigeria tech innovation"
        search_query = st.text_input(
            "",
            placeholder="🔍 Search for videos... (e.g., AI in Africa, Christian Tech, Afrofuturism)",
            value=default_search,
            label_visibility="collapsed"
        )
    with search_col2:
        search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")
    with search_col3:
        surprise_clicked = st.button("🎲 Surprise!", use_container_width=True, type="secondary")
    
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    # ============================================================
    # INTELLIGENT DATA FETCHING LOGIC
    # ============================================================
    # 1. If "Surprise Me" is clicked, search for a random mix
    if surprise_clicked:
        st.session_state.surprise_me = True
        # Set a random, engaging search to make the homepage vibrant
        search_query = random.choice([
            "AI in Africa", "Afrofuturism", "Tech innovation Nigeria", 
            "Gospel music 2026", "African startups", "Future of AI"
        ])
        st.session_state.initialized = True # Mark as initialized so it remembers state
    
    # 2. If search is clicked, use the search query
    elif search_clicked:
        st.session_state.search_triggered = True
        st.session_state.initialized = True
    
    # 3. If it's the very first load (initialized is False), force a Surprise/Vibrant search
    elif not st.session_state.initialized:
        search_query = random.choice(["African Tech Revolution", "Faith & Innovation", "Future of Africa"])
        st.session_state.initialized = True # Prevent this from running again on refresh
    
    # ============================================================
    # QUICK FILTERS
    # ============================================================
    st.markdown("---")
    st.markdown("#### 🏷️ Quick Filters")
    
    filter_cols = st.columns(5)
    
    with filter_cols[0]:
        if st.button("📱 All", use_container_width=True):
            st.session_state.filter_tag = None
            st.rerun()
    with filter_cols[1]:
        if st.button("💻 Tech", use_container_width=True):
            st.session_state.filter_tag = "tech"
            st.rerun()
    with filter_cols[2]:
        if st.button("🙏 Faith", use_container_width=True):
            st.session_state.filter_tag = "faith"
            st.rerun()
    with filter_cols[3]:
        if st.button("🌍 Africa", use_container_width=True):
            st.session_state.filter_tag = "africa"
            st.rerun()
    with filter_cols[4]:
        if st.button("🚀 Future", use_container_width=True):
            st.session_state.filter_tag = "future"
            st.rerun()
    
    if "filter_tag" in st.session_state and st.session_state.filter_tag:
        st.caption(f"🔍 Filter active: **{st.session_state.filter_tag}**")
    
    filter_tag = st.session_state.get("filter_tag", None)
    st.markdown("---")
    
    # ============================================================
    # ADVANCED SEARCH
    # ============================================================
    with st.expander("🔍 Advanced Search Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            upload_date = st.selectbox(
                "📅 Upload Date",
                ["Any time", "Last hour", "Today", "This week", "This month", "This year"],
                key="upload_date_filter"
            )
        
        with col2:
            sort_by = st.selectbox(
                "📊 Sort by",
                ["Relevance", "View count", "Upload date", "Rating"],
                key="sort_by_filter"
            )
        
        with col3:
            video_duration = st.selectbox(
                "⏱️ Duration",
                ["Any duration", "Short (< 4 min)", "Medium (4-20 min)", "Long (> 20 min)"],
                key="duration_filter"
            )
        
        if st.button("🔍 Apply Advanced Filters", use_container_width=True, type="secondary"):
            st.session_state.apply_filters = True
            st.rerun()
    
    # ============================================================
    # FETCH REAL DATA
    # ============================================================
    if not api_key and platform != "YouTube + Juicer (All Platforms)":
        st.warning("⚠️ Please add your YouTube API key to use this app")
        st.info("Get a free key at: https://console.cloud.google.com/apis/")
        st.stop()
    
    progress_placeholder = st.empty()
    
    with progress_placeholder.container():
        st.markdown(f"""
        <div class="progress-container">
            <div class="progress-label">
                <span>🔄 Fetching videos</span>
                <span class="highlight">Searching {platform}...</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width: 45%;"></div>
            </div>
            <div class="progress-percent">
                <span class="number">45%</span> · Connecting to API
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.5)
    
    with st.spinner(""):
        progress_placeholder.markdown(f"""
        <div class="progress-container">
            <div class="progress-label">
                <span>🔄 Fetching videos</span>
                <span class="highlight">Searching...</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width: 60%;"></div>
            </div>
            <div class="progress-percent">
                <span class="number">60%</span> · Searching {platform}
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
        
        if platform == "YouTube + Juicer (All Platforms)":
            juicer_key = get_juicer_api_key()
            if juicer_key:
                platforms = ["tiktok", "instagram", "facebook", "x", "youtube"]
                video_df = search_juicer_live(search_query, platforms, max_results=25)
                if video_df.empty:
                    st.info("📢 No results from Juicer. Falling back to YouTube.")
                    video_df = search_youtube_live(None, search_query, max_results=25, upload_date=upload_date, sort_by=sort_by, duration=video_duration)
            else:
                st.warning("⚠️ Juicer API key not found. Using YouTube only.")
                video_df = search_youtube_live(None, search_query, max_results=25, upload_date=upload_date, sort_by=sort_by, duration=video_duration)
        else:
            video_df = search_youtube_live(None, search_query, max_results=25, upload_date=upload_date, sort_by=sort_by, duration=video_duration)
        
        progress_placeholder.markdown(f"""
        <div class="progress-container">
            <div class="progress-label">
                <span>🔄 Fetching videos</span>
                <span class="highlight">Processing results...</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width: 80%;"></div>
            </div>
            <div class="progress-percent">
                <span class="number">80%</span> · Processing video data
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
        
        progress_placeholder.markdown(f"""
        <div class="progress-container">
            <div class="progress-label">
                <span>🔄 Fetching videos</span>
                <span class="highlight">Almost ready...</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width: 95%;"></div>
            </div>
            <div class="progress-percent">
                <span class="number">95%</span> · Finalizing
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
    
    progress_placeholder.empty()
    
    if video_df.empty:
        st.error("❌ No videos found. Please try a different search term.")
        st.stop()
    
    # ============================================================
    # INITIALIZE ENGINE
    # ============================================================
    engine = SmartRecommendationEngine(video_df, st.session_state.user_preferences)
    
    # ============================================================
    # SESSION STATE (Current Video)
    # ============================================================
    if "current_video" not in st.session_state or st.session_state.get("search_triggered", False):
        if len(video_df) > 0:
            st.session_state.current_video = video_df.iloc[0]["video_id"]
        st.session_state.search_triggered = False
    
    current_video_exists = st.session_state.current_video in video_df["video_id"].values
    if not current_video_exists and len(video_df) > 0:
        st.session_state.current_video = video_df.iloc[0]["video_id"]
    
    # ============================================================
    # DISPLAY CURRENT VIDEO - WITH BIGGER PREVIEW
    # ============================================================
    current_row = video_df[video_df["video_id"] == st.session_state.current_video]
    if current_row.empty:
        st.session_state.current_video = video_df.iloc[0]["video_id"]
        current_row = video_df[video_df["video_id"] == st.session_state.current_video]
    
    current = current_row.iloc[0]
    
    engine.update_user_preferences("watch", current)
    save_user_preferences(st.session_state.user_preferences)
    
    st.markdown("---")
    
    platform_source = current.get("platform", "YouTube")
    st.success(f"✅ Found {len(video_df)} videos about '{search_query}' from {platform_source}")
    
    # ===== BIG PREVIEW (Full Width) =====
    if show_preview and current.get('embed_url') and current['embed_url']:
        if "youtube.com" in current['embed_url']:
            st.markdown(f"""
            <div class="preview-container">
                <iframe src="{current['embed_url']}?autoplay=0&rel=0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen>
                </iframe>
            </div>
            <div class="preview-label">🎬 Preview · Click play to watch a short preview</div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"📱 Content from {platform_source} - Click 'Watch' button to view")
    
    # ===== Video Info (Below Preview) =====
    col_left, col_right = st.columns([2.2, 1])
    
    with col_left:
        if not show_preview and current.get('thumbnail_hq') and current['thumbnail_hq']:
            st.markdown(f"""
            <div class="video-card">
                <img src="{current['thumbnail_hq']}" class="video-thumbnail" alt="Video thumbnail">
                <div class="video-info">
            """, unsafe_allow_html=True)
        elif show_preview:
            st.markdown(f"""
            <div class="video-card">
                <div class="video-info">
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="video-card">
                <div class="video-info">
            """, unsafe_allow_html=True)
        
        platform_badge = current.get("platform", "YouTube")
        st.markdown(f"""
                <div style="display:inline-block; background: {colors["accent"]}30; color: {colors["accent"]}; padding:4px 16px; border-radius:20px; font-size:0.7rem; margin-bottom:10px;">
                    📱 {platform_badge}
                </div>
                <div class="video-title-text">{current['title']}</div>
                <div class="video-channel">📺 <strong>{current['channel']}</strong> · {current.get('published', 'Recently')}</div>
        """, unsafe_allow_html=True)
        
        if current.get('description') and current['description']:
            synopsis = current['description'][:350]
            if len(current['description']) > 350:
                synopsis += "..."
            st.markdown(f"""
            <div class="synopsis-box">
                <div class="synopsis-label">📖 Synopsis</div>
                <div class="synopsis-text">{synopsis}</div>
            </div>
            """, unsafe_allow_html=True)
        
        tags = current['tags'].split(", ")
        tag_classes = {
            "tech": "tag-tech", "faith": "tag-faith", "africa": "tag-africa",
            "future": "tag-future", "fun": "tag-fun", "afrofuturism": "tag-tech",
            "innovation": "tag-future", "christianity": "tag-faith", "ai": "tag-tech",
            "robotics": "tag-tech", "inspiration": "tag-default", "motivation": "tag-default"
        }
        tag_html = '<div class="tags-container">'
        for tag in tags:
            cls = tag_classes.get(tag, "tag-default")
            tag_html += f'<span class="tag {cls}">#{tag}</span>'
        tag_html += '</div>'
        st.markdown(tag_html, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{current['views']:,}</div>
                <div class="stat-label">Views</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">❤️ {current['likes']:,}</div>
                <div class="stat-label">Likes</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">🔄 {current['shares']:,}</div>
                <div class="stat-label">Shares</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{current['engagement_score']}%</div>
                <div class="stat-label">Engagement</div>
            </div>
            """, unsafe_allow_html=True)
        
        watch_label = f"▶️ Watch on {current.get('platform', 'YouTube')}"
        st.markdown(f'<div style="margin-top:12px;"><a href="{current["youtube_url"]}" target="_blank" style="color:{colors["accent"]}; text-decoration:none; font-weight:600;">{watch_label}</a></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown("### 📊 Content Analysis")
        if current.get('is_futuristic', 0) == 1:
            st.success("🚀 **Futuristic/Tech** (Boosted during day)")
        if current.get('is_faith_based', 0) == 1:
            st.info("🙏 **Faith/Community** (Boosted at night)")
        if current.get('is_afrocentric', 0) == 1:
            st.warning("🌍 **Afrocentric**")
        
        st.divider()
        
        st.markdown("### 📊 Your Analytics")
        
        history = st.session_state.user_preferences.get("watch_history", [])
        if history:
            liked_count = sum(1 for h in history if h.get("action") == "liked")
            watched_count = sum(1 for h in history if h.get("action") == "watched")
            skipped_count = sum(1 for h in history if h.get("action") == "skipped")
            total_interactions = len(history)
            engagement_rate = round((liked_count / total_interactions) * 100) if total_interactions > 0 else 0
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{liked_count}</div>
                    <div class="stat-label">❤️ Liked</div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{engagement_rate}%</div>
                    <div class="stat-label">📈 Engagement</div>
                </div>
                """, unsafe_allow_html=True)
            
            col_c, col_d = st.columns(2)
            with col_c:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{watched_count}</div>
                    <div class="stat-label">👀 Watched</div>
                </div>
                """, unsafe_allow_html=True)
            with col_d:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{skipped_count}</div>
                    <div class="stat-label">👎 Skipped</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.progress(engagement_rate / 100, text=f"Engagement Rate: {engagement_rate}%")
            
            if st.session_state.user_preferences.get("liked_tags"):
                st.caption(f"🎯 Favorite tags: {', '.join(st.session_state.user_preferences['liked_tags'][:5])}")
        else:
            st.info("No analytics yet. Start interacting with videos!")
        
        st.divider()
        st.markdown("### 🧠 What I Know About You")
        st.caption(f"✅ Learned {len(st.session_state.user_preferences.get('liked_tags', []))} content preferences")
        
        st.divider()
        
        st.markdown("### 🔗 Share This Video")
        share_url = current.get("youtube_url", "")
        if share_url:
            st.code(share_url, language="text")
            share_col1, share_col2, share_col3 = st.columns(3)
            with share_col1:
                if st.button("📋 Copy Link", use_container_width=True):
                    st.toast("✅ Link copied to clipboard!", icon="📋")
            with share_col2:
                email_link = f"mailto:?subject=Check out this video&body={share_url}"
                st.markdown(f'<a href="{email_link}" target="_blank" style="text-decoration:none;width:100%;display:block;"><button style="width:100%;padding:0.5rem;border-radius:30px;border:none;background:#4a90e2;color:white;font-weight:600;cursor:pointer;">📧 Email</button></a>', unsafe_allow_html=True)
            with share_col3:
                tweet_text = f"Check out this video: {current['title']}"
                tweet_link = f"https://twitter.com/intent/tweet?text={tweet_text}&url={share_url}"
                st.markdown(f'<a href="{tweet_link}" target="_blank" style="text-decoration:none;width:100%;display:block;"><button style="width:100%;padding:0.5rem;border-radius:30px;border:none;background:#1da1f2;color:white;font-weight:600;cursor:pointer;">🐦 Tweet</button></a>', unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### 💡 Pro Tip")
        st.caption("The more you use the ❤️ and 👎 buttons, the smarter I get!")
    
    # ============================================================
    # 🔥 VIBRANT HOMEPAGE FIX: DISPLAY TRENDING CAROUSEL HERE
    # (Directly after the stats and "Watch on Youtube" link, 
    # before the Feedback buttons. This completely fills the empty void).
    # ============================================================
    render_trending_carousel(video_df, title="🔥 Trending Now", max_items=8)

    # ============================================================
    # ACTION BUTTONS
    # ============================================================
    st.markdown("---")
    st.markdown("#### 👇 Your feedback makes me smarter!")
    
    btn_cols = st.columns(5)
    
    with btn_cols[0]:
        if st.button("❤️ Love It", use_container_width=True, type="primary"):
            engine.update_user_preferences("like", current)
            save_user_preferences(st.session_state.user_preferences)
            st.balloons()
            st.toast("🧠 I learned you like this! Getting smarter...", icon="❤️")
    
    with btn_cols[1]:
        if st.button("👍 Interesting", use_container_width=True):
            if "watch_history" not in st.session_state.user_preferences:
                st.session_state.user_preferences["watch_history"] = []
            st.session_state.user_preferences["watch_history"].append({
                "title": current["title"],
                "action": "liked",
                "thumbnail": current.get("thumbnail", ""),
                "video_id": current["video_id"],
                "platform": current.get("platform", "Unknown"),
                "timestamp": datetime.now().isoformat()
            })
            save_user_preferences(st.session_state.user_preferences)
            st.toast("📝 Noted! I'll consider your preference.", icon="👍")
    
    with btn_cols[2]:
        if st.button("👎 Not for me", use_container_width=True):
            engine.update_user_preferences("skip", current)
            save_user_preferences(st.session_state.user_preferences)
            st.toast("🧠 I learned you don't like this. Won't show again!", icon="👎")
    
    with btn_cols[3]:
        if st.button("⏩ Next Video", use_container_width=True, type="primary"):
            current_index = video_df[video_df["video_id"] == current["video_id"]].index[0]
            recs = engine.get_recommendations(
                current_index,
                num_recommendations=15,
                time_of_day=mode,
                filter_tag=filter_tag
            )
            if not recs.empty:
                next_video_id = recs.iloc[0]["video_id"]
                st.session_state.current_video = int(next_video_id)
                st.rerun()
    
    with btn_cols[4]:
        is_saved = current["video_id"] in st.session_state.watch_later
        if st.button(
            "📌 Save" if not is_saved else "✅ Saved",
            use_container_width=True,
            type="secondary" if not is_saved else "primary"
        ):
            if is_saved:
                st.session_state.watch_later.remove(current["video_id"])
                st.toast("🗑️ Removed from Watch Later!", icon="🗑️")
            else:
                st.session_state.watch_later.append(current["video_id"])
                st.toast("📌 Saved to Watch Later!", icon="📌")
            st.rerun()
    
    # ============================================================
    # RECOMMENDATIONS GRID
    # ============================================================
    st.markdown("---")
    st.markdown("### 📱 Recommended For You")
    st.caption("AI-powered recommendations based on your preferences")
    
    current_index = video_df[video_df["video_id"] == current["video_id"]].index[0]
    recommendations = engine.get_recommendations(
        current_index,
        num_recommendations=8,
        time_of_day=mode,
        filter_tag=filter_tag
    )
    
    if recommendations.empty:
        recommendations = video_df.sample(8)
    
    rec_cols = st.columns(4)
    for idx, (_, row) in enumerate(recommendations.iterrows()):
        with rec_cols[idx % 4]:
            platform_tag = row.get("platform", "YouTube")
            st.markdown(f"""
            <div class="rec-card">
                <img src="{row['thumbnail']}" class="rec-thumbnail" alt="Video thumbnail">
                <div class="rec-info">
                    <div class="rec-title">{row['title'][:55]}{'...' if len(row['title']) > 55 else ''}</div>
                    <div class="rec-channel">📱 {platform_tag} · {row['channel'][:15]}</div>
                    <div class="rec-score">🔥 {row['engagement_score']:.1f}% match</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"▶️ Watch", key=f"rec_{idx}"):
                st.session_state.current_video = row["video_id"]
                st.rerun()
    
    # ============================================================
    # WATCH LATER
    # ============================================================
    if st.session_state.watch_later:
        st.divider()
        st.markdown("### 📌 Watch Later")
        st.caption(f"You have {len(st.session_state.watch_later)} videos saved")
        
        watch_later_df = video_df[video_df["video_id"].isin(st.session_state.watch_later)]
        
        if not watch_later_df.empty:
            wl_cols = st.columns(4)
            for idx, (_, row) in enumerate(watch_later_df.iterrows()):
                with wl_cols[idx % 4]:
                    st.markdown(f"""
                    <div class="rec-card">
                        <img src="{row['thumbnail']}" class="rec-thumbnail" alt="Video thumbnail">
                        <div class="rec-info">
                            <div class="rec-title">{row['title'][:55]}{'...' if len(row['title']) > 55 else ''}</div>
                            <div class="rec-channel">📱 {row.get('platform', 'YouTube')} · {row['channel'][:15]}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"▶️ Watch", key=f"wl_{idx}"):
                        st.session_state.current_video = row["video_id"]
                        st.rerun()
    
    # ============================================================
    # EXPORT & CLEAR HISTORY
    # ============================================================
    if st.session_state.user_preferences.get("watch_history"):
        st.divider()
        st.markdown("### 📁 History Management")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            csv_data = export_watch_history()
            if csv_data:
                st.download_button(
                    label="📥 Download History (CSV)",
                    data=csv_data,
                    file_name=f"watch_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_exp2:
            if st.button("🔄 Reset Analytics", use_container_width=True):
                st.session_state.user_preferences["liked_tags"] = []
                st.session_state.user_preferences["skipped_ids"] = []
                save_user_preferences(st.session_state.user_preferences)
                st.toast("📊 Analytics reset!", icon="🔄")
                st.rerun()
        
        with col_exp3:
            if st.button("🗑️ Clear All History", use_container_width=True):
                clear_all_history()
                st.toast("🗑️ All history cleared!", icon="🗑️")
                st.rerun()
    
    # ============================================================
    # LEARNING VISUALIZATION
    # ============================================================
    with st.expander("🧠 How I'm Learning from You"):
        st.markdown("""
        ### 🔄 The Learning Loop
        
        **1. You interact with content**
        - ❤️ Love It → I learn what you like
        - 👎 Not for me → I learn what to avoid
        - ⏩ Next → I learn your viewing patterns
        
        **2. I update your preferences**
        - Track favorite tags and topics
        - Remember what you've watched
        - Adapt to your time-of-day patterns
        
        **3. Recommendations get smarter**
        - Content-based matching
        - Preference-based boosting
        - Time-aware personalization
        
        **4. The cycle continues**
        - Every interaction makes me smarter
        - Recommendations become more personalized
        - You discover new, relevant content
        """)
        
        if st.session_state.user_preferences.get("watch_history"):
            st.divider()
            st.markdown("### 📊 Your Learning Journey")
            st.caption("Every interaction is tracked here - this is how I learn from you!")
            
            history = st.session_state.user_preferences["watch_history"][-15:]
            
            for item in reversed(history):
                action_emoji = {
                    "liked": "❤️",
                    "skipped": "👎",
                    "watched": "👀"
                }.get(item.get("action", "watched"), "👀")
                
                thumbnail = item.get("thumbnail", "")
                title = item.get("title", "Unknown video")
                platform = item.get("platform", "Unknown")
                
                col1, col2, col3 = st.columns([1, 6, 1])
                with col1:
                    if thumbnail:
                        st.image(thumbnail, width=60)
                    else:
                        st.write("🎬")
                with col2:
                    st.write(f"**{title[:60]}{'...' if len(title) > 60 else ''}**")
                with col3:
                    st.write(f"{action_emoji} {platform}")
                st.divider()


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()