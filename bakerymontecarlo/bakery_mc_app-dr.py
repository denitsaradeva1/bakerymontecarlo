    seed = st.number_input("Seed", min_value=0, max_value=1000, value=42)
    capacity_gain_per_staff = st.number_input("Capacity gain per staff", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key="capacity_v7")
    demand_noise_sd = st.number_input("Demand noise (std dev)", min_value=0.0, max_value=0.5, value=0.10, step=0.02, key="noise_v7")
    safety = st.number_input("Safety factor", min_value=0.5, max_value=2.0, value=1.0, step=0.1, key="safety_v7")

# Production plan
# ...