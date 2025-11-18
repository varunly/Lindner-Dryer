def allocate_energy(e: pd.DataFrame, ivals: pd.DataFrame) -> pd.DataFrame:
    """
    Allocate energy consumption to products based on time overlap (MEMORY OPTIMIZED).
    
    Args:
        e: Parsed energy dataframe
        ivals: Exploded interval dataframe
        
    Returns:
        Dataframe with energy allocated to products
    """
    logger.info("Allocating energy to products...")
    results = []
    
    for zone_label, zone_name in ZONE_ENERGY_MAPPING.items():
        energy_col = f"E_{zone_name}_kWh"
        
        if energy_col not in e.columns:
            logger.warning(f"Energy column {energy_col} not found")
            continue
        
        # Filter energy records with non-zero values
        e_zone = e[e[energy_col].notna() & (e[energy_col] > 0)].copy()
        if e_zone.empty:
            logger.warning(f"No energy data for {zone_label}")
            continue
        
        # Filter intervals for this zone
        ivals_zone = ivals[ivals["Zone"] == zone_label].copy()
        if ivals_zone.empty:
            logger.warning(f"No intervals for {zone_label}")
            continue
        
        logger.info(f"Processing {zone_label}: {len(e_zone)} energy records × {len(ivals_zone)} intervals")
        
        # MEMORY-EFFICIENT: Process in chunks to avoid memory explosion
        chunk_size = 1000
        zone_results = []
        
        for i in range(0, len(ivals_zone), chunk_size):
            chunk = ivals_zone.iloc[i:i+chunk_size]
            
            # Cross join using dummy key (only for chunk)
            e_temp = e_zone.copy()
            chunk_temp = chunk.copy()
            
            e_temp['_key'] = 1
            chunk_temp['_key'] = 1
            
            merged = e_temp.merge(chunk_temp, on='_key', suffixes=('_e', '_p'))
            merged.drop('_key', axis=1, inplace=True)
            
            # Filter for overlapping time ranges
            merged = merged[
                (merged['P_end'] > merged['E_start']) & 
                (merged['P_start'] < merged['E_end'])
            ]
            
            if merged.empty:
                continue
            
            # Calculate overlap hours (vectorized)
            merged['latest_start'] = merged[['E_start', 'P_start']].max(axis=1)
            merged['earliest_end'] = merged[['E_end', 'P_end']].min(axis=1)
            merged['overlap_h'] = (
                (merged['earliest_end'] - merged['latest_start']).dt.total_seconds() / 3600
            ).clip(lower=0)
            
            # Filter zero overlaps
            merged = merged[merged['overlap_h'] > 0]
            
            if merged.empty:
                continue
            
            # Calculate energy share
            merged['Energy_share_kWh'] = merged[energy_col] * merged['overlap_h']
            
            # Select and rename columns
            result = merged[[
                'Month_e', 'Produkt', 'm3', 
                'Energy_share_kWh', 'overlap_h'
            ]].rename(columns={'Month_e': 'Month', 'overlap_h': 'Overlap_h'})
            
            result['Zone'] = zone_label
            zone_results.append(result)
        
        if zone_results:
            results.append(pd.concat(zone_results, ignore_index=True))
    
    if results:
        final_result = pd.concat(results, ignore_index=True)
        logger.info(f"Allocated {len(final_result)} energy records")
        return final_result
    else:
        logger.warning("No energy could be allocated")
        return pd.DataFrame(columns=[
            'Month', 'Zone', 'Produkt', 
            'Energy_share_kWh', 'Overlap_h', 'm3'
        ])
