import json
import logging
import os
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET

from app.config import settings
from app.models import EpisodeMetadata

logger = logging.getLogger(__name__)


class RSSGenerator:
    """
    Manages and publishes the static Podcast 2.0 RSS feed (feed.xml)
    with full iTunes namespace compliance and RFC 2822 dates for Apple Watch / Apple Podcasts.
    """

    def __init__(
        self,
        podcast_dir: Optional[Path] = None,
        base_url: Optional[str] = None
    ) -> None:
        self.podcast_dir = podcast_dir or settings.PODCAST_DIR
        self._custom_base_url = base_url
        self.feed_path = self.podcast_dir / "feed.xml"
        self.metadata_path = self.podcast_dir / "episodes.json"
        self.podcast_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_url(self) -> str:
        return (self._custom_base_url or settings.BASE_URL).rstrip("/")

    def refresh_episodes_and_regenerate(self) -> Path:
        """
        Update enclosure_url of all stored episodes to match current base_url and regenerate feed.xml.
        """
        episodes = self._load_episodes_metadata()
        for ep in episodes:
            ep.enclosure_url = f"{self.base_url}/podcast/episodes/{ep.filename}"
        self._save_episodes_metadata(episodes)
        return self.generate_feed(episodes)

    def _load_episodes_metadata(self) -> List[EpisodeMetadata]:
        """Load stored episodes metadata from JSON catalog."""
        if not self.metadata_path.exists():
            return []
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [
                    EpisodeMetadata(
                        guid=item["guid"],
                        title=item["title"],
                        description=item["description"],
                        filename=item["filename"],
                        file_size_bytes=item["file_size_bytes"],
                        duration_sec=item["duration_sec"],
                        pub_date=datetime.fromisoformat(item["pub_date"]),
                        enclosure_url=item["enclosure_url"]
                    )
                    for item in data
                ]
        except Exception as e:
            logger.error("Failed to load episodes metadata catalog: %s", e)
            return []

    def _save_episodes_metadata(self, episodes: List[EpisodeMetadata]) -> None:
        """Atomically persist episode metadata catalog to JSON."""
        data = [
            {
                "guid": ep.guid,
                "title": ep.title,
                "description": ep.description,
                "filename": ep.filename,
                "file_size_bytes": ep.file_size_bytes,
                "duration_sec": ep.duration_sec,
                "pub_date": ep.pub_date.isoformat(),
                "enclosure_url": ep.enclosure_url
            }
            for ep in episodes
        ]
        tmp_path = self.metadata_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.metadata_path)

    def add_episode(
        self,
        guid: str,
        title: str,
        description: str,
        filename: str,
        file_size_bytes: int,
        duration_sec: int,
        pub_date: Optional[datetime] = None
    ) -> EpisodeMetadata:
        """
        Record a new episode and regenerate feed.xml.
        """
        pub_dt = pub_date or datetime.now(timezone.utc)
        enclosure_url = f"{self.base_url}/podcast/episodes/{filename}"

        episode = EpisodeMetadata(
            guid=guid,
            title=title,
            description=description,
            filename=filename,
            file_size_bytes=file_size_bytes,
            duration_sec=duration_sec,
            pub_date=pub_dt,
            enclosure_url=enclosure_url
        )

        episodes = self._load_episodes_metadata()
        # Remove any existing item with the same guid or filename
        episodes = [e for e in episodes if e.guid != guid and e.filename != filename]
        # Insert latest at beginning (chronological descending order)
        episodes.insert(0, episode)

        self._save_episodes_metadata(episodes)
        self.generate_feed(episodes)
        return episode

    def generate_feed(self, episodes: Optional[List[EpisodeMetadata]] = None) -> Path:
        """
        Generate and atomically write feed.xml according to Podcast 2.0 / iTunes specs.
        """
        if episodes is None:
            episodes = self._load_episodes_metadata()

        # Sort episodes newest first
        episodes = sorted(episodes, key=lambda e: e.pub_date, reverse=True)

        # XML Namespaces
        itunes_ns = "http://www.itunes.com/dtds/podcast-1.0.dtd"
        content_ns = "http://purl.org/rss/1.0/modules/content/"
        atom_ns = "http://www.w3.org/2005/Atom"

        ET.register_namespace("itunes", itunes_ns)
        ET.register_namespace("content", content_ns)
        ET.register_namespace("atom", atom_ns)

        rss = ET.Element("rss", {
            "version": "2.0",
        })

        channel = ET.SubElement(rss, "channel")

        # Basic Channel Metadata
        ET.SubElement(channel, "title").text = settings.PODCAST_TITLE
        ET.SubElement(channel, "link").text = f"{self.base_url}/podcast/feed.xml"
        ET.SubElement(channel, "description").text = settings.PODCAST_DESCRIPTION
        ET.SubElement(channel, "language").text = settings.PODCAST_LANGUAGE
        ET.SubElement(channel, "generator").text = "running-coach-generator"

        # Atom Self Link (Mandatory for podcast aggregators)
        ET.SubElement(channel, f"{{{atom_ns}}}link", {
            "href": f"{self.base_url}/podcast/feed.xml",
            "rel": "self",
            "type": "application/rss+xml"
        })

        # iTunes Metadata
        ET.SubElement(channel, f"{{{itunes_ns}}}author").text = settings.PODCAST_AUTHOR
        ET.SubElement(channel, f"{{{itunes_ns}}}summary").text = settings.PODCAST_DESCRIPTION
        ET.SubElement(channel, f"{{{itunes_ns}}}explicit").text = "no"

        if settings.PODCAST_IMAGE_URL:
            ET.SubElement(channel, f"{{{itunes_ns}}}image", {"href": settings.PODCAST_IMAGE_URL})
            img = ET.SubElement(channel, "image")
            ET.SubElement(img, "url").text = settings.PODCAST_IMAGE_URL
            ET.SubElement(img, "title").text = settings.PODCAST_TITLE
            ET.SubElement(img, "link").text = f"{self.base_url}/podcast/feed.xml"

        # Category
        cat_health = ET.SubElement(channel, f"{{{itunes_ns}}}category", {"text": "Health & Fitness"})
        ET.SubElement(cat_health, f"{{{itunes_ns}}}category", {"text": "Running"})

        # Episode Items
        for ep in episodes:
            item = ET.SubElement(channel, "item")
            ET.SubElement(item, "title").text = ep.title
            ET.SubElement(item, "description").text = ep.description
            guid_elem = ET.SubElement(item, "guid", {"isPermaLink": "false"})
            guid_elem.text = ep.guid

            # RFC 2822 formatted date (mandatory for Apple Podcasts)
            ET.SubElement(item, "pubDate").text = format_datetime(ep.pub_date)

            # Enclosure for media file
            ET.SubElement(item, "enclosure", {
                "url": ep.enclosure_url,
                "length": str(ep.file_size_bytes),
                "type": "audio/mpeg"
            })

            ET.SubElement(item, f"{{{itunes_ns}}}duration").text = str(int(ep.duration_sec))
            ET.SubElement(item, f"{{{itunes_ns}}}explicit").text = "no"

        # Pretty-print indentation
        tree = ET.ElementTree(rss)
        ET.indent(tree, space="  ", level=0)

        # Atomic write
        tmp_feed = self.feed_path.with_suffix(".xml.tmp")
        tree.write(str(tmp_feed), encoding="utf-8", xml_declaration=True)
        tmp_feed.replace(self.feed_path)

        logger.info("Updated RSS feed with %d episodes at %s", len(episodes), self.feed_path)
        return self.feed_path

    def get_feed_xml(self) -> str:
        """Return the raw XML content of feed.xml."""
        if not self.feed_path.exists():
            self.generate_feed([])
        return self.feed_path.read_text(encoding="utf-8")
