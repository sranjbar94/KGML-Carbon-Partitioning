class KGML_SelfSup(nn.Module):
    """
    Knowledge-Guided ML (KGML) Self-Supervised Carbon Partitioning Model
    Reference: Ranjbar et al., 2025
    """
    def __init__(self, nfeat):
        super().__init__()

        # --- Shared Encoder ---
        self.enc = nn.Sequential(
            nn.Linear(nfeat, 128),
            nn.ReLU(),
            nn.Dropout(0.1),  # MC Dropout for uncertainty
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # --- Stomatal Conductance ---
        self.gs_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        # --- GPP Head ---
        self.gpp_head = nn.Sequential(
            nn.Linear(129, 64),  # 128 from encoder + 1 gs
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # --- Respiration Heads ---
        self.r_auto_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )
        self.r_het_head = nn.Sequential(
            nn.Linear(130, 32),  # 128 + Tsoil + SM
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # --- WUE Prior Estimator ---
        self.wue_hat = nn.Sequential(
            nn.Linear(4, 32),
            nn.Softplus(),
            nn.Linear(32, 1)
        )

        # --- Parameters ---
        self.coe_gpp_auto_abv0_raw = nn.Parameter(torch.tensor(0.0))
        self.alpha0 = nn.Parameter(torch.tensor(0.0))
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.beta_medlyn = nn.Parameter(torch.tensor(1.0))
        self.tlat_mult = nn.Parameter(torch.tensor(1.2))
        self.tlat_beta = 0.6

    def forward(self, X, idx_map):
        z = self.enc(X)

        # --- Extract Features ---
        PAR = X[:, [idx_map['PAR_NEON']]]
        SM = X[:, [idx_map['SoilMoisture_NEON']]]
        VPD = X[:, [idx_map['vpd_NEON']]]
        LE = X[:, [idx_map['LE']]]
        Tair = X[:, [idx_map['AirTemperature_NEON']]]
        Tsoil = X[:, [idx_map['T_soil1to4']]]
        SW_in = X[:, [idx_map['SW_IN']]]
        SW_out = X[:, [idx_map['SW_OUT']]]
        LW_in = X[:, [idx_map['LW_IN']]]
        LW_out = X[:, [idx_map['LW_OUT']]]
        H = X[:, [idx_map['H']]]
        G = X[:, [idx_map['G_1to5']]]
        NEE_PI = X[:, [idx_map['NEE_PI']]]
        Pfvs_obs = X[:, [idx_map['Pfvs_NEON']]]
        Rfvs_obs = X[:, [idx_map['Rfvs_NEON']]]
        Tfvs_obs = X[:, [idx_map['Tfvs_NEON']]]

        # --- Stomatal Conductance ---
        gs = torch.clamp(self.gs_net(z), min=0.0)

        # --- GPP ---
        GPP = F.softplus(self.gpp_head(torch.cat([z, gs], -1)))
        GPP = torch.clamp(GPP, min=0.0)

        # --- Respiration ---
        coe_gpp_auto_abv0 = 0.05 + 0.68 * torch.sigmoid(self.coe_gpp_auto_abv0_raw)
        R_auto_above, R_auto_below = F.softplus(self.r_auto_head(z)).chunk(2, dim=-1)
        R_auto_above += coe_gpp_auto_abv0 * GPP
        R_auto = R_auto_above + R_auto_below
        R_het = F.softplus(self.r_het_head(torch.cat([z, Tsoil, SM], -1)))
        R_het *= torch.exp(torch.log(torch.tensor(2.0)) * (Tsoil - 15.0)/10.0)
        RECO = R_auto + R_het

        # --- Latent ET ---
        eps = 1e-5
        T_lat = F.softplus(self.tlat_mult * gs * (VPD + eps), beta=self.tlat_beta)
        ET_lat = T_lat + 0.05 * F.softplus(PAR)

        # --- WUE ---
        WUE_hat = self.wue_hat(torch.cat([VPD, Tfvs_obs, Tair, SM], -1))

        # --- ET from LE ---
        ET_from_LE = self.alpha0 + self.alpha1 * (LE / 2.45e6)

        # --- NEE ---
        NEE_pred = RECO - GPP
        Pfvs_pred = GPP - R_auto_above
        Rfvs_pred = R_auto_below + R_het

        # --- Energy Balance ---
        Rn = (SW_in - SW_out) + (LW_in - LW_out)
        L_heat = ET_lat
        L_energy_balance = Rn - (L_heat + H + G)
        L_energy = torch.mean(torch.abs(L_energy_balance) / torch.clamp(torch.abs(Rn), min=1e-6))

        return {
            'GPP': GPP, 'RECO': RECO, 'R_auto': R_auto, 'R_auto_above': R_auto_above,
            'R_auto_below': R_auto_below, 'R_hetero': R_het, 'gs': gs, 'T_lat': T_lat,
            'ET_lat': ET_lat, 'WUE_hat': WUE_hat, 'ET_LE': ET_from_LE, 'NEE_pred': NEE_pred,
            'Pfvs_pred': Pfvs_pred, 'Rfvs_pred': Rfvs_pred, 'L_energy_balance': L_energy_balance
        }
