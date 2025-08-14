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
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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
        
        # Setup headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def setup_selenium_driver(self) -> webdriver.Chrome:
        """Setup Selenium WebDriver with appropriate options."""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
        
        return webdriver.Chrome(options=chrome_options)
    
    async def collect_virtual_football_data(self) -> List[Dict[str, Any]]:
        """Collect virtual football game data."""
        url = f"{self.base_url}/virtual-football"
        games_data = []
        
        try:
            driver = self.setup_selenium_driver()
            driver.get(url)
            
            # Wait for games to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "virtual-game"))
            )
            
            # Extract game data
            games = driver.find_elements(By.CLASS_NAME, "virtual-game")
            
            for game in games:
                try:
                    game_data = {
                        'timestamp': datetime.now(),
                        'game_type': 'virtual_football',
                        'teams': self._extract_teams(game),
                        'odds': self._extract_odds(game),
                        'game_time': self._extract_game_time(game),
                        'league': self._extract_league(game),
                        'match_id': self._extract_match_id(game)
                    }
                    games_data.append(game_data)
                except Exception as e:
                    self.logger.warning(f"Error extracting game data: {e}")
            
            driver.quit()
            
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
    
    def _extract_teams(self, game_element) -> Dict[str, str]:
        """Extract team names from game element."""
        try:
            teams = game_element.find_elements(By.CLASS_NAME, "team-name")
            return {
                'home': teams[0].text.strip() if len(teams) > 0 else '',
                'away': teams[1].text.strip() if len(teams) > 1 else ''
            }
        except:
            return {'home': '', 'away': ''}
    
    def _extract_odds(self, game_element) -> Dict[str, float]:
        """Extract odds from game element."""
        try:
            odds_elements = game_element.find_elements(By.CLASS_NAME, "odds")
            return {
                'home_win': float(odds_elements[0].text.strip()) if len(odds_elements) > 0 else 0.0,
                'draw': float(odds_elements[1].text.strip()) if len(odds_elements) > 1 else 0.0,
                'away_win': float(odds_elements[2].text.strip()) if len(odds_elements) > 2 else 0.0
            }
        except:
            return {'home_win': 0.0, 'draw': 0.0, 'away_win': 0.0}
    
    def _extract_game_time(self, game_element) -> str:
        """Extract game time from element."""
        try:
            time_element = game_element.find_element(By.CLASS_NAME, "game-time")
            return time_element.text.strip()
        except:
            return ''
    
    def _extract_league(self, game_element) -> str:
        """Extract league name from element."""
        try:
            league_element = game_element.find_element(By.CLASS_NAME, "league-name")
            return league_element.text.strip()
        except:
            return ''
    
    def _extract_match_id(self, game_element) -> str:
        """Extract unique match ID."""
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
    
    def close(self):
        """Clean up resources."""
        self.session.close()
        self.db_manager.close()