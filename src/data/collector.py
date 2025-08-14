"""
Betika Virtual Games Data Collector

This module handles data collection from Betika's virtual games platform.
Supports multiple virtual game types and includes rate limiting and error handling.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

import aiohttp
import requests
from bs4 import BeautifulSoup
import pandas as pd
from playwright.async_api import async_playwright, Browser, Page

from ..utils.database import DatabaseManager
from ..utils.config import Config


class BetikaDataCollector:
    """Collects data from Betika virtual games platform."""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.data_collection['base_url']
        self.virtual_games = config.data_collection['virtual_games']
        self.session = requests.Session()
        self.db_manager = DatabaseManager(config)
        self.logger = logging.getLogger(__name__)
        self.playwright = None
        self.browser = None
        
        # Setup headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    async def setup_playwright_browser(self) -> Browser:
        """Setup Playwright browser with appropriate options."""
        if not self.playwright:
            self.playwright = await async_playwright().start()
        
        if not self.browser:
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-blink-features=AutomationControlled'
                ]
            )
        
        return self.browser
    
    async def collect_virtual_football_data(self) -> List[Dict[str, Any]]:
        """Collect virtual football game data."""
        url = f"{self.base_url}/virtual-football"
        games_data = []
        
        try:
            browser = await self.setup_playwright_browser()
            page = await browser.new_page()
            
            # Set extra headers and user agent
            await page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            # Navigate to the page
            await page.goto(url, wait_until='domcontentloaded', timeout=30000)
            
            # Wait for games to load
            try:
                await page.wait_for_selector('.virtual-game, .game-item, .match-item', timeout=10000)
            except:
                # Try alternative selectors if main ones don't exist
                await page.wait_for_load_state('networkidle', timeout=5000)
            
            # Extract game data using multiple possible selectors
            game_selectors = ['.virtual-game', '.game-item', '.match-item', '.bet-item']
            games = []
            
            for selector in game_selectors:
                try:
                    games = await page.query_selector_all(selector)
                    if games:
                        break
                except:
                    continue
            
            for game in games:
                try:
                    game_data = {
                        'timestamp': datetime.now(),
                        'game_type': 'virtual_football',
                        'teams': await self._extract_teams_playwright(game),
                        'odds': await self._extract_odds_playwright(game),
                        'game_time': await self._extract_game_time_playwright(game),
                        'league': await self._extract_league_playwright(game),
                        'match_id': await self._extract_match_id_playwright(game)
                    }
                    games_data.append(game_data)
                except Exception as e:
                    self.logger.warning(f"Error extracting game data: {e}")
            
            await page.close()
            
        except Exception as e:
            self.logger.error(f"Error collecting virtual football data: {e}")
        
        return games_data
    
    async def collect_virtual_basketball_data(self) -> List[Dict[str, Any]]:
        """Collect virtual basketball game data."""
        url = f"{self.base_url}/virtual-basketball"
        games_data = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Parse basketball games
                        games = soup.find_all('div', class_='basketball-game')
                        
                        for game in games:
                            game_data = {
                                'timestamp': datetime.now(),
                                'game_type': 'virtual_basketball',
                                'teams': self._extract_basketball_teams(game),
                                'odds': self._extract_basketball_odds(game),
                                'game_time': self._extract_basketball_time(game),
                                'league': self._extract_basketball_league(game)
                            }
                            games_data.append(game_data)
                            
        except Exception as e:
            self.logger.error(f"Error collecting virtual basketball data: {e}")
        
        return games_data
    
    async def _extract_teams_playwright(self, game_element) -> Dict[str, str]:
        """Extract team names from game element using Playwright."""
        try:
            # Try multiple possible selectors for team names
            team_selectors = ['.team-name', '.team', '.home-team', '.away-team', '.participant']
            teams = []
            
            for selector in team_selectors:
                try:
                    team_elements = await game_element.query_selector_all(selector)
                    if team_elements:
                        teams = [await elem.text_content() for elem in team_elements]
                        break
                except:
                    continue
            
            # Clean and return team names
            clean_teams = [team.strip() for team in teams if team and team.strip()]
            
            return {
                'home': clean_teams[0] if len(clean_teams) > 0 else '',
                'away': clean_teams[1] if len(clean_teams) > 1 else ''
            }
        except:
            return {'home': '', 'away': ''}
    
    def _extract_teams(self, game_element) -> Dict[str, str]:
        """Extract team names from game element (legacy method)."""
        try:
            teams = game_element.find_elements(By.CLASS_NAME, "team-name")
            return {
                'home': teams[0].text.strip() if len(teams) > 0 else '',
                'away': teams[1].text.strip() if len(teams) > 1 else ''
            }
        except:
            return {'home': '', 'away': ''}
    
    async def _extract_odds_playwright(self, game_element) -> Dict[str, float]:
        """Extract odds from game element using Playwright."""
        try:
            # Try multiple possible selectors for odds
            odds_selectors = ['.odds', '.odd', '.price', '.coefficient', '.bet-odd']
            odds = []
            
            for selector in odds_selectors:
                try:
                    odds_elements = await game_element.query_selector_all(selector)
                    if odds_elements:
                        odds_texts = [await elem.text_content() for elem in odds_elements]
                        odds = [self._parse_odds_value(text) for text in odds_texts if text]
                        break
                except:
                    continue
            
            return {
                'home_win': odds[0] if len(odds) > 0 else 0.0,
                'draw': odds[1] if len(odds) > 1 else 0.0,
                'away_win': odds[2] if len(odds) > 2 else odds[1] if len(odds) > 1 else 0.0
            }
        except:
            return {'home_win': 0.0, 'draw': 0.0, 'away_win': 0.0}
    
    def _parse_odds_value(self, text: str) -> float:
        """Parse odds value from text, handling various formats."""
        try:
            if not text:
                return 0.0
            # Remove common non-numeric characters
            clean_text = text.strip().replace(',', '.').replace(' ', '')
            # Extract first number found
            import re
            match = re.search(r'\d+\.?\d*', clean_text)
            return float(match.group()) if match else 0.0
        except:
            return 0.0
    
    def _extract_odds(self, game_element) -> Dict[str, float]:
        """Extract odds from game element (legacy method)."""
        try:
            odds_elements = game_element.find_elements(By.CLASS_NAME, "odds")
            return {
                'home_win': float(odds_elements[0].text.strip()) if len(odds_elements) > 0 else 0.0,
                'draw': float(odds_elements[1].text.strip()) if len(odds_elements) > 1 else 0.0,
                'away_win': float(odds_elements[2].text.strip()) if len(odds_elements) > 2 else 0.0
            }
        except:
            return {'home_win': 0.0, 'draw': 0.0, 'away_win': 0.0}
    
    async def _extract_game_time_playwright(self, game_element) -> str:
        """Extract game time from element using Playwright."""
        try:
            time_selectors = ['.game-time', '.time', '.match-time', '.start-time']
            for selector in time_selectors:
                try:
                    time_element = await game_element.query_selector(selector)
                    if time_element:
                        text = await time_element.text_content()
                        return text.strip() if text else ''
                except:
                    continue
            return ''
        except:
            return ''
    
    async def _extract_league_playwright(self, game_element) -> str:
        """Extract league name from element using Playwright."""
        try:
            league_selectors = ['.league-name', '.league', '.competition', '.tournament']
            for selector in league_selectors:
                try:
                    league_element = await game_element.query_selector(selector)
                    if league_element:
                        text = await league_element.text_content()
                        return text.strip() if text else ''
                except:
                    continue
            return ''
        except:
            return ''
    
    async def _extract_match_id_playwright(self, game_element) -> str:
        """Extract unique match ID using Playwright."""
        try:
            # Try different attribute names for match ID
            id_attributes = ['data-match-id', 'data-id', 'id', 'data-game-id']
            for attr in id_attributes:
                try:
                    match_id = await game_element.get_attribute(attr)
                    if match_id:
                        return match_id
                except:
                    continue
            return ''
        except:
            return ''
    
    def _extract_game_time(self, game_element) -> str:
        """Extract game time from element (legacy method)."""
        try:
            time_element = game_element.find_element(By.CLASS_NAME, "game-time")
            return time_element.text.strip()
        except:
            return ''
    
    def _extract_league(self, game_element) -> str:
        """Extract league name from element (legacy method)."""
        try:
            league_element = game_element.find_element(By.CLASS_NAME, "league-name")
            return league_element.text.strip()
        except:
            return ''
    
    def _extract_match_id(self, game_element) -> str:
        """Extract unique match ID (legacy method)."""
        try:
            return game_element.get_attribute('data-match-id') or ''
        except:
            return ''
    
    def _extract_basketball_teams(self, game_element) -> Dict[str, str]:
        """Extract basketball team names."""
        try:
            teams = game_element.find_all('span', class_='team')
            return {
                'home': teams[0].text.strip() if len(teams) > 0 else '',
                'away': teams[1].text.strip() if len(teams) > 1 else ''
            }
        except:
            return {'home': '', 'away': ''}
    
    def _extract_basketball_odds(self, game_element) -> Dict[str, float]:
        """Extract basketball odds."""
        try:
            odds = game_element.find_all('span', class_='odd-value')
            return {
                'home_win': float(odds[0].text.strip()) if len(odds) > 0 else 0.0,
                'away_win': float(odds[1].text.strip()) if len(odds) > 1 else 0.0
            }
        except:
            return {'home_win': 0.0, 'away_win': 0.0}
    
    def _extract_basketball_time(self, game_element) -> str:
        """Extract basketball game time."""
        try:
            time_elem = game_element.find('span', class_='game-time')
            return time_elem.text.strip() if time_elem else ''
        except:
            return ''
    
    def _extract_basketball_league(self, game_element) -> str:
        """Extract basketball league."""
        try:
            league_elem = game_element.find('span', class_='league')
            return league_elem.text.strip() if league_elem else ''
        except:
            return ''
    
    async def collect_all_games(self) -> Dict[str, List[Dict]]:
        """Collect data from all configured virtual games."""
        all_data = {}
        
        # Collect from different game types
        if 'virtual-football' in self.virtual_games:
            all_data['football'] = await self.collect_virtual_football_data()
        
        if 'virtual-basketball' in self.virtual_games:
            all_data['basketball'] = await self.collect_virtual_basketball_data()
        
        # Save to database
        for game_type, games in all_data.items():
            if games:
                self.db_manager.save_games_data(games, game_type)
                self.logger.info(f"Saved {len(games)} {game_type} games to database")
        
        return all_data
    
    async def start_continuous_collection(self):
        """Start continuous data collection based on configured interval."""
        interval = self.config.data_collection['collection_interval']
        
        self.logger.info(f"Starting continuous data collection every {interval} seconds")
        
        while True:
            try:
                start_time = time.time()
                data = await self.collect_all_games()
                
                total_games = sum(len(games) for games in data.values())
                self.logger.info(f"Collected {total_games} games in {time.time() - start_time:.2f} seconds")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in continuous collection: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def get_historical_data(self, game_type: str, days: int = 30) -> pd.DataFrame:
        """Retrieve historical data from database."""
        return self.db_manager.get_historical_data(game_type, days)
    
    async def close(self):
        """Clean up resources."""
        try:
            if self.browser:
                await self.browser.close()
                self.browser = None
            
            if self.playwright:
                await self.playwright.stop()
                self.playwright = None
        except Exception as e:
            self.logger.warning(f"Error closing Playwright resources: {e}")
        
        self.session.close()
        self.db_manager.close()