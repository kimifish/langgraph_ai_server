# pyright: basic
# pyright: reportAttributeAccessIssue=false

from datetime import datetime
import os
import requests
import logging
from langchain_core.tools import tool
from typing import List
import time
from mpd import MPDClient
from urllib.parse import quote
import paramiko
from io import BytesIO
from pathlib import Path

from ai_server.config import cfg, APP_NAME
log = logging.getLogger(f'{APP_NAME}.{__name__}')


def _upload_playlist_via_ssh(playlist_name: str, content: str, host: str) -> str:
    """
    Uploads playlist file to MPD server via SSH.
    
    Args:
        playlist_name: Name of the playlist file
        content: Content of the playlist
        host: Hostname from config
        
    Returns:
        Error message or empty string on success
        
    Raises:
        Exception: If SSH/SFTP operations fail
    """
    # Upload to MPD server via SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # Load system SSH configuration
    ssh_config = paramiko.SSHConfig()
    user_config_file = os.path.expanduser(f"~/.ssh/config.d/{host.replace('.lan', '')}")
    if os.path.exists(user_config_file):
        with open(user_config_file) as f:
            ssh_config.parse(f)
    
    # Get host config
    host_config = ssh_config.lookup(host.replace(".lan", ""))
    log.debug(f'{host_config=}')
    
    ssh.connect(
        hostname=host_config.get('hostname', host),
        username=host_config.get('user', 'pi'),
        key_filename=host_config.get('identityfile', str([Path.home() / '.ssh' / 'lan'][0])),
        port=int(host_config.get('port', 22))
    )
    
    # Create SFTP session
    sftp = ssh.open_sftp()
    
    # Convert content to bytes with explicit encoding
    content_bytes = content.encode('utf-8')
    
    # Create BytesIO object
    playlist_file = BytesIO(content_bytes)
    
    # Upload the file
    remote_path = f"{cfg.music.mpd.playlists_path}/{playlist_name}"
    sftp.putfo(playlist_file, remote_path)
    
    # Close everything
    playlist_file.close()
    sftp.close()
    ssh.close()
    
    return ""

@tool(parse_docstring=True)
def get_music_by_tags(tags: str, limit: int = 30, distance: float = 1.2) -> str:
    """
    Queries music database for tracks, creates M3U playlist and uploads it to MPD server.

    Args:
        tags: Comma-separated list of music tags/descriptions. Tags only in english, artist names - as in original.
              Example - "rock, guitar solo, energetic, Сплин" or "ambient, calm, meditation"
        limit: Maximum number of tracks to return (default: 30)
        distance: Maximum similarity distance between tags (default: 1.2)
    
    Returns:
        Name of the created playlist (without path) or error message.
        If error occurs, do not retry unless specifically mentioned.
    """
    try:
        # Prepare the URL with encoded tags
        encoded_tags = quote(tags)
        url = f"{cfg.music.music_db.host}:{cfg.music.music_db.port}{cfg.music.music_db.endpoint}/?tags={encoded_tags}&limit={limit}&max_distance={distance}"
        
        # Get tracks from music database
        response = requests.get(url)
        response.raise_for_status()
        tracks = response.json()
        
        if not tracks:
            return "No tracks found matching these tags. Try different tags or descriptions."

        # Create playlist name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_tags = tags.replace(',', '').replace(' ', '_')[:20]  # First 20 chars of tags without spaces and commas
        playlist_name = f"ai_{safe_tags}_{timestamp}.m3u"
        
        # Create M3U content with explicit line endings
        m3u_content = "#EXTM3U\n" + "\n".join(tracks) + "\n"
        
        try:
            error = _upload_playlist_via_ssh(playlist_name, m3u_content, cfg.music.mpd.host)
            if error:
                log.error(f"SSH upload error: {error}")
                return "Music system is currently unavailable. Please try again in a few minutes. Do not retry immediately."
                
            log.info(f"Created playlist {playlist_name} with {len(tracks)} tracks")
            return playlist_name

        except Exception as e:
            log.error(f"SSH/SFTP error: {e}")
            return "Music system is currently unreachable. This likely requires administrator attention. Please try again later or use other music-related commands."
            
    except requests.exceptions.RequestException as e:
        log.error(f"Error querying music database: {e}")
        return "Music database is currently unavailable. This is likely a temporary issue. Please try again in a few minutes. Do not retry immediately."
    except Exception as e:
        log.error(f"Unexpected error while creating playlist: {e}")
        return "An unexpected error occurred with the music system. This requires administrator attention. Please try other music-related commands instead."

# @tool(parse_docstring=True)
def create_playlist(songs: List[str], name: str = "") -> str:
    """
    Creates a playlist from the given songs.

    Args:
        songs: List of song filenames to add to playlist
        name: Optional playlist name. If empty, generates timestamp-based name.
    
    Returns:
        Playlist name or error message
    """
    try:
        playlist_name = name or f"playlist_{int(time.time())}"
        # Add your playlist creation logic here
        return f"Created playlist '{playlist_name}' with {len(songs)} songs"
    except Exception as e:
        log.error(f"Error creating playlist: {e}")
        return f"Failed to create playlist: {str(e)}"

@tool(parse_docstring=True)
def play_playlist(playlist_name: str) -> str:
    """
    Plays a playlist on the MPD server.

    Args:
        playlist_name: Name of the playlist to play

    Returns:
        Status message indicating success or failure
    """
    try:
        client = MPDClient()
        client.connect(cfg.music.mpd.host, cfg.music.mpd.port)
        
        # Handle password if configured
        if hasattr(cfg.music.mpd, 'password'):
            if cfg.music.mpd.password == '.env':
                password = os.getenv('MPD_PASSWORD')
                if password:
                    client.password(password)
            else:
                client.password(cfg.music.mpd.password)

        # Clear current playlist
        client.clear()
        
        # Load and play the playlist
        client.load(playlist_name.replace(".m3u", ""))
        client.play()
        
        # Get current song info
        current = client.currentsong()
        client.close()
        client.disconnect()
        
        return f"Playing playlist '{playlist_name}'. Now playing: {current.get('title', 'Unknown')}"

    except Exception as e:
        log.error(f"Error playing playlist on MPD: {e}")
        return f"Failed to play playlist: {str(e)}"

@tool(parse_docstring=True)
def mpd_control(command: str) -> str:
    """
    Controls MPD playback.

    Args:
        command: One of: play, pause, stop, next, previous, shuffle, clear
    
    Returns:
        Status message
    """
    try:
        client = MPDClient()
        client.connect(cfg.music.mpd.host, cfg.music.mpd.port)
        
        # Handle password if configured
        if hasattr(cfg.music.mpd, 'password'):
            if cfg.music.mpd.password == '.env':
                password = os.getenv('MPD_PASSWORD')
                if password:
                    client.password(password)
            else:
                client.password(cfg.music.mpd.password)

        command = command.lower()
        if command == 'play':
            client.play()
            status = "Playing"
        elif command == 'pause':
            client.pause()
            status = "Paused"
        elif command == 'stop':
            client.stop()
            status = "Stopped"
        elif command == 'next':
            client.next()
            status = "Skipped to next track"
        elif command == 'previous':
            client.previous()
            status = "Returned to previous track"
        elif command == 'shuffle':
            client.shuffle()
            status = "Playlist shuffled"
        elif command == 'clear':
            client.clear()
            status = "Playlist cleared"
        else:
            status = f"Unknown command: {command}"

        # Get current song info if playing
        if command in ['play', 'next', 'previous']:
            current = client.currentsong()
            if current:
                status += f". Now playing: {current.get('title', 'Unknown')}"

        client.close()
        client.disconnect()
        return status

    except Exception as e:
        log.error(f"Error controlling MPD: {e}")
        return f"Failed to execute command: {str(e)}"
