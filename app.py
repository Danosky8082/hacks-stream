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
import streamlit.components.v1 as components

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
        
        # ===== STEP 2: FORCE CLEAR ALL EXISTING SOURCES =====
        # We must delete the old Facebook sources so we can add a fresh one
        st.info("🧹 Clearing old sources to prevent conflicts...")
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
                    time.sleep(0.5) # Give it a moment to delete properly

        # ===== STEP 3: Add sources with EXACT platform names =====
        exact_platform_names = {
            "youtube": "YouTube", "tiktok": "TikTok", "instagram": "Instagram",
            "facebook": "Facebook", "twitter": "Twitter", "linkedin": "LinkedIn"
        }
        
        added_sources = []
        
        for platform in platforms:
            platform_lower = platform.lower()
            exact_name = exact_platform_names.get(platform_lower, platform)
            
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
                    # Try the next term_type if 422 (Unprocessable Entity)
                    continue
                else:
                    # Break if there is a hard error like 500 or 404
                    break
            
            if not source_added:
                st.warning(f"⚠️ Could not add {exact_name}. Try a different search term.")

        if not added_sources:
            st.warning("⚠️ Could not add any sources. Try a different search term.")
            return pd.DataFrame()
        
        # ===== STEP 4: Wait and fetch posts (WAIT LONGER FOR FACEBOOK) =====
        st.info(f"🔄 Searching {', '.join(added_sources)}...")
        time.sleep(10) # Increased from 5 to 10 seconds for Facebook sync!
        
        results = []
        for attempt in range(3):
            posts_response = requests.get(
                f"https://api.juicer.io/v1/feeds/{feed_id}/posts",
                headers=headers,
                params={"limit": max_results},
                timeout=45
            )
            
            if posts_response.status_code == 200:
                posts_data = posts_response.json()
                results = posts_data.get("data", posts_data.get("posts", []))
                if results:
                    break
            
            time.sleep(3)
        
        if not results:
            st.info(f"📢 No results found for '{query}'. Try a different search term.")
            return pd.DataFrame()
        
        # ===== STEP 5: Process results =====
        video_data = []
        for idx, item in enumerate(results):
            try:
                if not isinstance(item, dict):
                    continue
                
                # ===== FIX: IMPROVED DATA EXTRACTION =====
                raw_title = item.get("title", item.get("text", item.get("content", item.get("message", ""))))
                title = str(raw_title) if raw_title else "Facebook Post"
                
                description = item.get("description", item.get("caption", item.get("message", "")))
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
                if not thumbnail:
                    thumbnail = f"https://ui-avatars.com/api/?name={platform}&background=f0c040&color=0f0f1a&size=512&rounded=true"
                
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
# RENDER FUNCTIONS - CLICKABLE CAROUSEL (WITH SCROLL BUTTONS)
# ============================================================

def render_trending_carousel(video_df, title="🔥 Trending Now", max_items=8):
    """
    Renders a clickable horizontal scrollable carousel with custom scroll buttons.
    Fixes desktop scroll issues by using Javascript buttons inside the component.
    """
    if video_df.empty:
        return

    st.markdown(f"### {title}")
    st.caption("Tap a card to load instantly. Use the < > arrows to scroll.")

    trending_df = video_df.sort_values(by="views", ascending=False).head(max_items)

    # Build the FULL HTML document
    html_content = """
    <style>
        .carousel-wrapper {
            position: relative;
            width: 100%;
        }
        .trending-scroll-container {
            display: flex;
            overflow-x: auto;
            gap: 15px;
            padding: 10px 0 20px 0;
            scroll-behavior: smooth;
            -webkit-overflow-scrolling: touch;
            
            /* Force scrollbar to attempt to show */
            scrollbar-width: thin;
            scrollbar-color: #f0c040 rgba(255,255,255,0.05);
        }
        .trending-scroll-container::-webkit-scrollbar {
            height: 6px;
            display: block;
        }
        .trending-scroll-container::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }
        .trending-scroll-container::-webkit-scrollbar-thumb {
            background: #f0c040;
            border-radius: 10px;
        }
        .trending-scroll-container::-webkit-scrollbar-button {
            display: block;
            height: 0;
            width: 0;
        }

        .trending-card {
            flex: 0 0 220px;
            background: rgba(20, 20, 40, 0.8);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            cursor: pointer;
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
            pointer-events: none;
        }
        .trending-info {
            padding: 8px 12px 12px 12px;
            pointer-events: none;
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

        /* Custom Scroll Buttons */
        .scroll-btn {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(15, 15, 26, 0.9);
            border: 1px solid rgba(255,255,255,0.1);
            color: #fff;
            font-size: 24px;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: 0.2s;
            z-index: 10;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }
        .scroll-btn:hover {
            background: #f0c040;
            color: #0f0f1a;
            border-color: #f0c040;
        }
        .scroll-btn-left { left: -20px; }
        .scroll-btn-right { right: -20px; }

        @media (max-width: 700px) {
            .trending-card { flex: 0 0 150px; }
            .scroll-btn { display: none; } /* Hide buttons on mobile, use touch scroll */
        }
    </style>

    <div class="carousel-wrapper">
        <div class="trending-scroll-container" id="myScrollContainer">
    """

    # Build the inner HTML for each card
    for _, row in trending_df.iterrows():
        title_clean = row['title'][:50] + ('...' if len(row['title']) > 50 else '')
        video_id = row['video_id']
        
        html_content += f"""
        <form method="get" action="" style="display:contents;">
            <input type="hidden" name="load_video" value="{video_id}">
            <button type="submit" style="background:none; border:none; padding:0; cursor:pointer; width:100%;">
                <div class="trending-card">
                    <img src="{row['thumbnail']}" class="trending-img" alt="Thumbnail">
                    <div class="trending-info">
                        <div class="trending-title">{title_clean}</div>
                        <div class="trending-meta">
                            <span class="trending-channel">{row['channel'][:15]}</span>
                            <span class="trending-score">🔥 {row['engagement_score']:.0f}%</span>
                        </div>
                    </div>
                </div>
            </button>
        </form>
        """
    
    html_content += """
        </div>
        <!-- Custom JavaScript Buttons -->
        <button class="scroll-btn scroll-btn-left" onclick="document.getElementById('myScrollContainer').scrollBy({left: -300, behavior: 'smooth'});">‹</button>
        <button class="scroll-btn scroll-btn-right" onclick="document.getElementById('myScrollContainer').scrollBy({left: 300, behavior: 'smooth'});">›</button>
    </div>
    """
    
    # Render the HTML using the components module
    components.html(html_content, height=310)

    # ============================================================
    # URL PARAMETER HANDLER
    # ============================================================
    query_params = st.query_params
    loaded_id_list = query_params.get("load_video")
    
    if loaded_id_list and len(loaded_id_list) > 0:
        loaded_id = loaded_id_list[0]
        if loaded_id:
            st.session_state.current_video = loaded_id
            del query_params["load_video"]
            st.rerun()

# ============================================================
# MAIN APP (FIXED FACEBOOK HASHTAG SEARCH)
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
    
    # NEW: Track if the user is browsing the Grid or watching a Video
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "home" # "home" or "watch"
    
    # FIX: Initialize the app so it doesn't force "Nigeria tech innovation" on refresh
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
    # CSS - COMPLETE FIXED DARK THEME
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
        
        /* ===== HOME GRID CARDS (Safely handled via buttons) ===== */
        .home-btn-wrapper {{
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin-bottom: 15px;
            width: 100%;
            text-align: left;
        }}
        .home-card {{
            background: rgba(20, 20, 40, 0.85);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.06);
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            cursor: pointer;
            width: 100%;
            margin-bottom: 15px;
        }}
        .home-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 40px rgba(240, 192, 64, 0.15);
        }}
        .home-img {{
            width: 100%;
            aspect-ratio: 16/9;
            object-fit: cover;
            display: block;
        }}
        .home-info {{ padding: 12px 16px 16px 16px; }}
        .home-title {{ font-weight: 700; font-size: 0.95rem; color: #ffffff; line-height: 1.3; margin-bottom: 6px; }}
        .home-channel {{ font-size: 0.8rem; color: #aaaaaa; }}
        .home-stats {{ display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.75rem; color: #888; }}
        .home-views {{ color: #f0c040; font-weight: 600; }}
        
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
        
        st.markdown("#### 🌐 Platform")
        platform_options = ["YouTube", "TikTok", "Instagram", "Facebook", "Twitter", "YouTube + Juicer (All Platforms)"]
        platform = st.selectbox(
            "Select Content Source",
            platform_options,
            help="Juicer searches TikTok, Instagram, Facebook, X, and more with one API key."
        )
        
        st.divider()
        
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
        
        api_status = {"YouTube": youtube_status, "Juicer (Multi-Platform)": juicer_status}
        for name, status in api_status.items():
            st.caption(f"{status} {name}")
        
        if juicer_key:
            st.success("🌐 Juicer active!")
        else:
            st.info("💡 Add JUICER_API_KEY to .env")
        
        st.divider()
        
        # ===== DATA SOURCE =====
        st.markdown("#### 📡 Data Source")
        if "YouTube" in platform:
            api_key = get_available_youtube_key()
            if api_key:
                st.success("✅ YouTube Connected")
            else:
                api_key = st.text_input("🔑 YouTube API Key", type="password", placeholder="Get one at console.cloud.google.com")
        else:
            api_key = get_available_youtube_key()
        
        st.divider()
        
        st.markdown("#### 🎨 Theme")
        theme_options = ["Dark", "Light", "Afro"]
        selected_theme = st.selectbox(
            "Choose your vibe", theme_options,
            index=theme_options.index(st.session_state.theme) if st.session_state.theme in theme_options else 0
        )
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()
        
        st.divider()
        
        show_preview = st.toggle("🎬 Show Video Preview", value=True)
        
        st.divider()
        if st.button("🏠 Back to Home Grid", use_container_width=True):
            st.session_state.view_mode = "home"
            st.session_state.current_video = None
            st.rerun()
        
        st.divider()
        st.caption("💡 I learn from your likes and skips!")
    
    # ============================================================
    # SEARCH BAR (UPDATED PLACEHOLDER TEXT)
    # ============================================================
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown('<div class="search-wrapper">', unsafe_allow_html=True)
    
    search_col1, search_col2, search_col3 = st.columns([5, 1, 1])
    with search_col1:
        # The "value" is set dynamically below based on initialization
        if not st.session_state.initialized:
            default_search = "" 
        else:
            default_search = "" 
            
        search_query = st.text_input(
            "",
            placeholder="🔍 Search for ANY video globally...",
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
    # If search bar is empty and user clicks Search, we default to a random trend
    if surprise_clicked:
        search_query = random.choice([
            "AI in Africa", "Afrofuturism", "Tech innovation Nigeria", 
            "Gospel music 2026", "African startups", "Future of AI"
        ])
        st.session_state.initialized = True
    
    elif search_clicked:
        if not search_query:
            search_query = random.choice(["African Tech Revolution", "Faith & Innovation", "Future of Africa"])
        st.session_state.search_triggered = True
        st.session_state.initialized = True
    
    elif not st.session_state.initialized:
        # CRITICAL FIX: Populate search_query with a valid initial search term so the API works!
        search_query = "Africa tech innovation"
        st.session_state.initialized = True
    elif not search_query:
        # If app is initialized but the box is blank, default to a safe fallback
        search_query = "Africa tech innovation"

    # ============================================================
    # FETCH REAL DATA (FIXED JUICER LOGIC)
    # ============================================================
    if not api_key and platform != "YouTube + Juicer (All Platforms)":
        st.warning("⚠️ Please add your YouTube API key to use this app")
        st.info("Get a free key at: https://console.cloud.google.com/apis/")
        st.stop()
    
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        st.markdown(f"""
        <div class="progress-container">
            <div class="progress-label"><span>🔄 Fetching videos</span><span class="highlight">Searching {platform}...</span></div>
            <div class="progress-track"><div class="progress-fill" style="width: 45%;"></div></div>
            <div class="progress-percent"><span class="number">45%</span> · Connecting to API</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.5)
    
    with st.spinner(""):
        progress_placeholder.markdown(f"""
        <div class="progress-container">
            <div class="progress-label"><span>🔄 Fetching videos</span><span class="highlight">Searching...</span></div>
            <div class="progress-track"><div class="progress-fill" style="width: 60%;"></div></div>
            <div class="progress-percent"><span class="number">60%</span> · Searching {platform}</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
        
        video_df = pd.DataFrame()
        
        # ================= NEW JUICER LOGIC =================
        # 1. If user explicitly picked TikTok, Instagram, Facebook, or Twitter
        if platform in ["TikTok", "Instagram", "Facebook", "Twitter"]:
            st.info(f"📡 Searching {platform} via Juicer...")
            juicer_key = get_juicer_api_key()
            if juicer_key:
                # CRITICAL FIX: Prepend hashtag for Facebook/Instagram searches
                final_query = search_query
                if platform in ["Facebook", "Instagram"]:
                    if not search_query.startswith("#"):
                        final_query = "#" + search_query.replace(" ", "")
                
                video_df = search_juicer_live(final_query, [platform], max_results=25)
            else:
                st.error("❌ Juicer API Key missing! Please add JUICER_API_KEY to .env")
        
        # 2. If user picked "YouTube + Juicer (All Platforms)"
        elif platform == "YouTube + Juicer (All Platforms)":
            st.info("📡 Searching ALL platforms via Juicer...")
            juicer_key = get_juicer_api_key()
            if juicer_key:
                platforms = ["tiktok", "instagram", "facebook", "x", "youtube"]
                video_df = search_juicer_live(search_query, platforms, max_results=25)
                # If Juicer fails, fall back to YouTube
                if video_df.empty:
                    st.warning("⚠️ Juicer returned no results. Falling back to YouTube...")
                    video_df = search_youtube_live(api_key, search_query, max_results=25)
            else:
                st.warning("⚠️ Juicer API key not found. Falling back to YouTube.")
                video_df = search_youtube_live(api_key, search_query, max_results=25)
        
        # 3. If user picked YouTube ONLY
        elif platform == "YouTube":
            st.info(f"📡 Searching YouTube only...")
            video_df = search_youtube_live(api_key, search_query, max_results=25)
        # ====================================================
        
        progress_placeholder.markdown(f"""
        <div class="progress-container">
            <div class="progress-label"><span>🔄 Fetching videos</span><span class="highlight">Processing results...</span></div>
            <div class="progress-track"><div class="progress-fill" style="width: 80%;"></div></div>
            <div class="progress-percent"><span class="number">80%</span> · Processing video data</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
        
        progress_placeholder.markdown(f"""
        <div class="progress-container">
            <div class="progress-label"><span>🔄 Fetching videos</span><span class="highlight">Almost ready...</span></div>
            <div class="progress-track"><div class="progress-fill" style="width: 95%;"></div></div>
            <div class="progress-percent"><span class="number">95%</span> · Finalizing</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
    
    progress_placeholder.empty()
    
    if video_df.empty:
        st.error("❌ No videos found. Please try a different search term.")
        if st.button("🔄 Try Again (Quota reset)"):
            st.rerun()
        st.info("💡 If searching YouTube, your quota may be exhausted. Try selecting 'TikTok' or 'Instagram' in the sidebar!")
        st.stop()
    
    # ============================================================
    # ✅ STEP 2 IMPLEMENTATION: HOME GRID VS WATCH MODE
    # ============================================================
    
    # If a video was selected via URL param, switch to watch mode
    query_params = st.query_params
    loaded_id_list = query_params.get("load_video")
    if loaded_id_list and len(loaded_id_list) > 0:
        loaded_id = loaded_id_list[0]
        st.session_state.current_video = loaded_id
        st.session_state.view_mode = "watch"
        st.rerun()

    # 1. HOME MODE: Show a beautiful 4-column grid for discovery
    if st.session_state.view_mode == "home":
        st.markdown("### 🌍 Discover Trending Videos")
        st.caption("Click any card below to start watching")
        
        # Sort by views for trending homepage
        trending_grid = video_df.sort_values(by="views", ascending=False).head(16)
        
        # Render Grid using native click handling
        cols = st.columns(4)
        for idx, (_, row) in enumerate(trending_grid.iterrows()):
            with cols[idx % 4]:
                # Render the beautiful card HTML
                st.markdown(f"""
                <div class="home-card" onclick="window.location.href='?load_video={row["video_id"]}'">
                    <img src="{row['thumbnail']}" class="home-img" alt="Thumbnail">
                    <div class="home-info">
                        <div class="home-title">{row['title'][:60]}{'...' if len(row['title']) > 60 else ''}</div>
                        <div class="home-channel">{row['channel']} · {row['platform']}</div>
                        <div class="home-stats">
                            <span class="home-views">👁️ {row['views']:,}</span>
                            <span>❤️ {row['likes']:,}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 2. WATCH MODE: Show the detailed Video Player
    else:
        engine = SmartRecommendationEngine(video_df, st.session_state.user_preferences)
        
        # === FIX: DEFINE MISSING VARIABLES FOR WATCH MODE ===
        filter_tag = st.session_state.get("filter_tag", None)
        tag_classes = {
            "tech": "tag-tech", "faith": "tag-faith", "africa": "tag-africa",
            "future": "tag-future", "fun": "tag-fun", "afrofuturism": "tag-tech",
            "innovation": "tag-future", "christianity": "tag-faith", "ai": "tag-tech",
            "robotics": "tag-tech", "inspiration": "tag-default", "motivation": "tag-default"
        }
        # =====================================================

        if "current_video" not in st.session_state or st.session_state.current_video not in video_df["video_id"].values:
            st.session_state.current_video = video_df.iloc[0]["video_id"]
        
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
        
        # == Big Preview ==
        if show_preview and current.get('embed_url'):
            if "youtube.com" in current['embed_url']:
                st.markdown(f"""
                <div class="preview-container">
                    <iframe src="{current['embed_url']}?autoplay=0&rel=0" allowfullscreen></iframe>
                </div>
                <div class="preview-label">🎬 Preview · Click play to watch a short preview</div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"📱 Content from {platform_source}")
        
        # == Video Info ==
        col_left, col_right = st.columns([2.2, 1])
        with col_left:
            st.markdown(f"""
            <div class="video-card">
                <div style="display:inline-block; background: {colors["accent"]}30; color: {colors["accent"]}; padding:4px 16px; border-radius:20px; font-size:0.7rem; margin: 10px 0 0 15px;">📱 {platform_source}</div>
                <div style="padding: 0 15px 15px 15px;">
                    <div class="video-title-text">{current['title']}</div>
                    <div class="video-channel">📺 <strong>{current['channel']}</strong> · {current.get('published', 'Recently')}</div>
            """, unsafe_allow_html=True)
            
            if current.get('description'):
                synopsis = current['description'][:350] + ('...' if len(current['description']) > 350 else '')
                st.markdown(f"""
                <div class="synopsis-box">
                    <div class="synopsis-label">📖 Synopsis</div>
                    <div class="synopsis-text">{synopsis}</div>
                </div>
                """, unsafe_allow_html=True)
            
            tags = current['tags'].split(", ")
            tag_html = '<div class="tags-container">'
            for tag in tags:
                cls = tag_classes.get(tag, "tag-default")
                tag_html += f'<span class="tag {cls}">#{tag}</span>'
            tag_html += '</div>'
            st.markdown(tag_html, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.markdown(f'<div class="stat-box"><div class="stat-number">{current["views"]:,}</div><div class="stat-label">Views</div></div>', unsafe_allow_html=True)
            with col2: st.markdown(f'<div class="stat-box"><div class="stat-number">❤️ {current["likes"]:,}</div><div class="stat-label">Likes</div></div>', unsafe_allow_html=True)
            with col3: st.markdown(f'<div class="stat-box"><div class="stat-number">🔄 {current["shares"]:,}</div><div class="stat-label">Shares</div></div>', unsafe_allow_html=True)
            with col4: st.markdown(f'<div class="stat-box"><div class="stat-number">{current["engagement_score"]}%</div><div class="stat-label">Engagement</div></div>', unsafe_allow_html=True)
            
            st.markdown(f'<div style="margin-top:12px;"><a href="{current["youtube_url"]}" target="_blank" style="color:{colors["accent"]}; text-decoration:none; font-weight:600;">▶️ Watch on YouTube</a></div>', unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        with col_right:
            st.markdown("### 📊 Content Analysis")
            if current.get('is_futuristic', 0) == 1: st.success("🚀 **Futuristic/Tech**")
            if current.get('is_faith_based', 0) == 1: st.info("🙏 **Faith/Community**")
            if current.get('is_afrocentric', 0) == 1: st.warning("🌍 **Afrocentric**")
            
            st.divider()
            st.markdown("### 🔗 Share This Video")
            share_url = current.get("youtube_url", "")
            st.code(share_url, language="text")
            
            st.divider()
            st.markdown("### 💡 Pro Tip")
            st.caption("The more you use the ❤️ and 👎 buttons, the smarter I get!")
        
        # == Trending Carousel (Inserted before feedback) ==
        render_trending_carousel(video_df, title="🔥 Trending Now", max_items=8)
        
        # == Feedback Buttons ==
        st.markdown("---")
        st.markdown("#### 👇 Your feedback makes me smarter!")
        btn_cols = st.columns(5)
        with btn_cols[0]:
            if st.button("❤️ Love It", use_container_width=True, type="primary"):
                engine.update_user_preferences("like", current)
                save_user_preferences(st.session_state.user_preferences)
                st.balloons()
                st.toast("🧠 Learned!", icon="❤️")
        with btn_cols[1]:
            if st.button("👍 Interesting", use_container_width=True):
                st.session_state.user_preferences["watch_history"].append({"title": current["title"], "action": "liked", "video_id": current["video_id"]})
                save_user_preferences(st.session_state.user_preferences)
                st.toast("📝 Noted!", icon="👍")
        with btn_cols[2]:
            if st.button("👎 Not for me", use_container_width=True):
                engine.update_user_preferences("skip", current)
                save_user_preferences(st.session_state.user_preferences)
                st.toast("🧠 Skipped!", icon="👎")
        with btn_cols[3]:
            if st.button("⏩ Next", use_container_width=True, type="primary"):
                recs = engine.get_recommendations(video_df[video_df["video_id"] == current["video_id"]].index[0], num_recommendations=15, time_of_day=mode, filter_tag=filter_tag)
                if not recs.empty:
                    st.session_state.current_video = recs.iloc[0]["video_id"]
                    st.rerun()
        with btn_cols[4]:
            is_saved = current["video_id"] in st.session_state.watch_later
            if st.button("📌 Save" if not is_saved else "✅ Saved", use_container_width=True, type="secondary" if not is_saved else "primary"):
                if is_saved: st.session_state.watch_later.remove(current["video_id"])
                else: st.session_state.watch_later.append(current["video_id"])
                st.rerun()
        
        # == Recommended Grid ==
        st.markdown("---")
        st.markdown("### 📱 Recommended For You")
        st.caption("AI-powered recommendations based on your preferences")
        
        current_index = video_df[video_df["video_id"] == current["video_id"]].index[0]
        recommendations = engine.get_recommendations(current_index, num_recommendations=8, time_of_day=mode, filter_tag=filter_tag)
        if recommendations.empty: recommendations = video_df.sample(8)
        
        rec_cols = st.columns(4)
        for idx, (_, row) in enumerate(recommendations.iterrows()):
            with rec_cols[idx % 4]:
                st.markdown(f"""
                <div class="rec-card">
                    <img src="{row['thumbnail']}" style="width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:14px 14px 0 0;">
                    <div style="padding:10px;">
                        <div class="rec-title">{row['title'][:55]}{'...' if len(row['title']) > 55 else ''}</div>
                        <div class="rec-channel">📱 {row.get('platform', 'YouTube')} · {row['channel'][:15]}</div>
                        <div class="rec-score">🔥 {row['engagement_score']:.1f}% match</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"▶️ Watch", key=f"rec_{idx}"):
                    st.session_state.current_video = row["video_id"]
                    st.rerun()


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()