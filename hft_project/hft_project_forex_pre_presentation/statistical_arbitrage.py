"""
Module for statistical arbitrage strategy implementation.
Generates trading signals based on cointegration analysis and manages risk parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('StatisticalArbitrage')

class StatArbitrageStrategy:
    """Statistical arbitrage strategy based on cointegration."""
    
    def __init__(
        self,
        z_score_entry: float = 2.0,
        z_score_exit: float = 0.0,
        max_z_score: float = 4.0,
        stop_loss_z: float = 4.0,
        max_positions: int = 5,
        risk_per_trade: float = 0.02
    ):
        """
        Initialize the statistical arbitrage strategy.
        
        Args:
            z_score_entry: Z-score threshold for entry (default: 2.0)
            z_score_exit: Z-score threshold for exit (default: 0.0)
            max_z_score: Maximum allowed z-score (default: 4.0)
            stop_loss_z: Z-score based stop loss (default: 4.0)
            max_positions: Maximum concurrent positions (default: 5)
            risk_per_trade: Risk per trade as fraction of capital (default: 0.02)
        """
        self.z_score_entry = z_score_entry
        self.z_score_exit = z_score_exit
        self.max_z_score = max_z_score
        self.stop_loss_z = stop_loss_z
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        
        # Active positions
        self.positions = []
        
        logger.info(f"Initialized StatArbitrageStrategy with z_score_entry={z_score_entry}, "
                   f"z_score_exit={z_score_exit}, max_positions={max_positions}")

    def generate_signals(self, cointegration_results: List[Dict]) -> List[Dict]:
        """
        Generate trading signals from cointegration analysis results.
        
        Args:
            cointegration_results: List of cointegration analysis results
            
        Returns:
            List of trading signals
        """
        logger.info(f"Generating signals from {len(cointegration_results)} cointegration results")
        
        signals = []
        
        for result in cointegration_results:
            pair1 = result['pair1']
            pair2 = result['pair2']
            z_score = result['current_z_score']
            hedge_ratio = result['hedge_ratio']
            
            # Skip if z-score is beyond max threshold
            if abs(z_score) > self.max_z_score:
                logger.warning(f"Skipping {pair1}/{pair2} - Z-score ({z_score}) exceeds maximum ({self.max_z_score})")
                continue
            
            # Generate entry signals
            if z_score > self.z_score_entry:
                # Spread is too wide (positive z-score) - expect reversion
                # Short pair1, long pair2
                signal = {
                    'pair1': pair1,
                    'pair2': pair2,
                    'action1': 'SELL',
                    'action2': 'BUY',
                    'ratio': hedge_ratio,
                    'z_score': z_score,
                    'signal_type': 'ENTRY',
                    'signal_time': datetime.now().isoformat(),
                    'signal_strength': abs(z_score) - self.z_score_entry,
                    'stop_loss_z': z_score + self.stop_loss_z
                }
                signals.append(signal)
                logger.info(f"Generated SELL/BUY signal for {pair1}/{pair2} (z-score: {z_score:.2f})")
                
            elif z_score < -self.z_score_entry:
                # Spread is too narrow (negative z-score) - expect widening
                # Long pair1, short pair2
                signal = {
                    'pair1': pair1,
                    'pair2': pair2,
                    'action1': 'BUY',
                    'action2': 'SELL',
                    'ratio': hedge_ratio,
                    'z_score': z_score,
                    'signal_type': 'ENTRY',
                    'signal_time': datetime.now().isoformat(),
                    'signal_strength': abs(z_score) - self.z_score_entry,
                    'stop_loss_z': z_score - self.stop_loss_z
                }
                signals.append(signal)
                logger.info(f"Generated BUY/SELL signal for {pair1}/{pair2} (z-score: {z_score:.2f})")
            
            # Generate exit signals for existing positions
            for position in self.positions:
                if position['pair1'] == pair1 and position['pair2'] == pair2:
                    # Check if position should be closed
                    if (position['action1'] == 'BUY' and z_score >= self.z_score_exit) or \
                       (position['action1'] == 'SELL' and z_score <= self.z_score_exit):
                        
                        exit_signal = {
                            'pair1': pair1,
                            'pair2': pair2,
                            'action1': 'SELL' if position['action1'] == 'BUY' else 'BUY',
                            'action2': 'BUY' if position['action2'] == 'SELL' else 'SELL',
                            'ratio': hedge_ratio,
                            'z_score': z_score,
                            'signal_type': 'EXIT',
                            'signal_time': datetime.now().isoformat(),
                            'entry_signal_id': position.get('signal_id'),
                            'profit_loss': self._calculate_profit_loss(position, z_score)
                        }
                        signals.append(exit_signal)
                        logger.info(f"Generated EXIT signal for {pair1}/{pair2} (z-score: {z_score:.2f})")
        
        # Sort signals by signal strength
        signals.sort(key=lambda x: x.get('signal_strength', 0), reverse=True)
        
        # Limit to max positions for entry signals
        entry_signals = [s for s in signals if s['signal_type'] == 'ENTRY']
        if len(entry_signals) > self.max_positions:
            entry_signals = entry_signals[:self.max_positions]
            signals = [s for s in signals if s['signal_type'] == 'EXIT'] + entry_signals
        
        logger.info(f"Generated {len(signals)} total signals ({len(entry_signals)} entry, "
                  f"{len(signals) - len(entry_signals)} exit)")
        return signals
    
    def _calculate_profit_loss(self, position: Dict, current_z_score: float) -> float:
        """
        Calculate estimated profit/loss for a position.
        
        Args:
            position: Position dictionary
            current_z_score: Current z-score
            
        Returns:
            Estimated profit/loss
        """
        # Simplified P&L calculation based on z-score movement
        entry_z = position['z_score']
        
        if position['action1'] == 'BUY':  # Long pair1, short pair2
            return entry_z - current_z_score
        else:  # Short pair1, long pair2
            return current_z_score - entry_z
    
    def calculate_position_sizes(
        self, 
        signal: Dict, 
        account_size: float, 
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Calculate optimal position sizes based on risk parameters.
        
        Args:
            signal: Trading signal
            account_size: Account size in base currency
            price_data: Dictionary of price data for currency pairs
            
        Returns:
            Dictionary with position size information
        """
        logger.info(f"Calculating position sizes for {signal['pair1']}/{signal['pair2']}")
        
        # Get current prices
        pair1_price = price_data[signal['pair1']]['close'].iloc[-1]
        pair2_price = price_data[signal['pair2']]['close'].iloc[-1]
        
        # Calculate volatility (using 20-day standard deviation)
        pair1_vol = price_data[signal['pair1']]['close'].pct_change().rolling(20).std().iloc[-1]
        pair2_vol = price_data[signal['pair2']]['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Risk amount
        risk_amount = account_size * self.risk_per_trade
        
        # Calculate position sizes based on volatility and hedge ratio
        # This is a simplified approach - real implementation would account for more factors
        pair1_units = risk_amount / (pair1_price * pair1_vol)
        pair2_units = pair1_units * signal['ratio'] * (pair1_price / pair2_price)
        
        # Round to whole units
        pair1_units = round(pair1_units)
        pair2_units = round(pair2_units)
        
        # Calculate notional values
        pair1_notional = pair1_units * pair1_price
        pair2_notional = pair2_units * pair2_price
        
        position_info = {
            'pair1_units': pair1_units,
            'pair2_units': pair2_units,
            'pair1_notional': pair1_notional,
            'pair2_notional': pair2_notional,
            'total_notional': pair1_notional + pair2_notional,
            'risk_amount': risk_amount
        }
        
        logger.info(f"Position sizes: {pair1_units} units of {signal['pair1']}, "
                  f"{pair2_units} units of {signal['pair2']}")
        return position_info
    
    def update_positions(self, signals: List[Dict], executed_signals: List[Dict]):
        """
        Update the list of active positions based on executed signals.
        
        Args:
            signals: List of generated signals
            executed_signals: List of signals that were executed
        """
        # Add new positions
        for signal in executed_signals:
            if signal['signal_type'] == 'ENTRY':
                self.positions.append(signal)
                logger.info(f"Added new position: {signal['pair1']}/{signal['pair2']}")
            
            elif signal['signal_type'] == 'EXIT':
                # Remove closed positions
                self.positions = [
                    p for p in self.positions 
                    if not (p['pair1'] == signal['pair1'] and p['pair2'] == signal['pair2'])
                ]
                logger.info(f"Closed position: {signal['pair1']}/{signal['pair2']}")
        
        logger.info(f"Active positions: {len(self.positions)}")
    
    def get_portfolio_status(self) -> Dict:
        """
        Get current portfolio status summary.
        
        Returns:
            Dictionary with portfolio statistics
        """
        return {
            'active_positions': len(self.positions),
            'positions': self.positions,
            'max_positions': self.max_positions,
            'risk_per_trade': self.risk_per_trade
        }