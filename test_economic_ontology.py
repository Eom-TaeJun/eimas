#!/usr/bin/env python3
"""Test Economic Ontology Implementation - Direct Import"""

def main():
    print("=" * 60)
    print("Economic Ontology Test")
    print("=" * 60)
    
    # Import directly from the module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "shock_propagation_graph", 
        "lib/shock_propagation_graph.py"
    )
    spg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spg)
    
    # 1. Test TRANSMISSION_TEMPLATES
    print(f"\n1. Templates: {len(spg.TRANSMISSION_TEMPLATES)} entries")
    for (source, target), data in list(spg.TRANSMISSION_TEMPLATES.items())[:5]:
        print(f"   {source} -> {target}: {data['sign']} [{data['theory_reference']}]")
    
    # 2. Test EconomicEdge
    print("\n2. EconomicEdge creation:")
    edge = spg.get_economic_edge("DFF", "SPY", lag=5, p_value=0.01)
    print(f"   Arrow: {edge.to_arrow()}")
    print(f"   Mechanism: {edge.mechanism}")
    print(f"   Theory: {edge.theory_reference}")
    print(f"   Narrative: {edge.narrative}")
    
    # 3. Test Narrative Generation
    print("\n3. Shock Narrative:")
    path = ["DFF", "M2", "SPY"]
    edges = [
        spg.get_economic_edge("DFF", "M2", lag=1),
        spg.get_economic_edge("M2", "SPY", lag=3)
    ]
    narrative = spg.generate_shock_narrative(path, edges)
    print(narrative)
    
    # 4. Test Impulse Response Text
    print("\n4. Impulse Response Text:")
    affected = {"SPY": -0.05, "QQQ": -0.08, "TLT": 0.03}
    text = spg.generate_impulse_response_text("DFF", 0.0025, affected)
    print(text[:500])
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
