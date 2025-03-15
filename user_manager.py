"""
Module for tracking user preferences and history.
"""
import os
import json
from typing import Dict, List, Any
from datetime import datetime

class UserManager:
    """
    Class for tracking user preferences and history.
    """
    def __init__(self, storage_file: str = "user_data.json"):
        """
        Initialize the UserManager with a storage file.
        
        Args:
            storage_file: Path to the JSON file for storing user data.
        """
        self.storage_file = storage_file
        self.user_data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """
        Load user data from the storage file.
        
        Returns:
            Dictionary of user data.
        """
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # Return default data if the file is corrupted
                return self._get_default_data()
        else:
            # Return default data if the file doesn't exist
            return self._get_default_data()
    
    def _get_default_data(self) -> Dict[str, Any]:
        """
        Get default user data structure.
        
        Returns:
            Dictionary of default user data.
        """
        return {
            "preferences": {
                "interests": [],
                "summary_type": "brief"
            },
            "history": []
        }
    
    def _save_data(self) -> None:
        """
        Save user data to the storage file.
        """
        with open(self.storage_file, "w") as f:
            json.dump(self.user_data, f, indent=2)
    
    def get_preferences(self) -> Dict[str, Any]:
        """
        Get user preferences.
        
        Returns:
            Dictionary of user preferences.
        """
        return self.user_data.get("preferences", {})
    
    def update_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Update user preferences.
        
        Args:
            preferences: Dictionary of preferences to update.
        """
        # Update existing preferences
        self.user_data["preferences"].update(preferences)
        self._save_data()
    
    def add_interest(self, interest: str) -> None:
        """
        Add a topic of interest.
        
        Args:
            interest: Topic of interest to add.
        """
        interests = self.user_data["preferences"].get("interests", [])
        
        # Add the interest if it's not already in the list
        if interest not in interests:
            interests.append(interest)
            self.user_data["preferences"]["interests"] = interests
            self._save_data()
    
    def remove_interest(self, interest: str) -> None:
        """
        Remove a topic of interest.
        
        Args:
            interest: Topic of interest to remove.
        """
        interests = self.user_data["preferences"].get("interests", [])
        
        # Remove the interest if it's in the list
        if interest in interests:
            interests.remove(interest)
            self.user_data["preferences"]["interests"] = interests
            self._save_data()
    
    def add_to_history(self, topic: str, summary_type: str) -> None:
        """
        Add a search to the history.
        
        Args:
            topic: Topic of the search.
            summary_type: Type of summary generated.
        """
        # Create the history item
        history_item = {
            "topic": topic,
            "summary_type": summary_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add the history item
        self.user_data["history"].append(history_item)
        
        # Keep only the last 20 items
        if len(self.user_data["history"]) > 20:
            self.user_data["history"] = self.user_data["history"][-20:]
        
        self._save_data()
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get search history.
        
        Args:
            limit: Maximum number of history items to return.
            
        Returns:
            List of history items.
        """
        # Get the history items
        history = self.user_data.get("history", [])
        
        # Sort by timestamp in descending order
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Return the limited history
        return history[:limit]
    
    def clear_history(self) -> None:
        """
        Clear search history.
        """
        self.user_data["history"] = []
        self._save_data()