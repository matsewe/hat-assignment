import gspread
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def _preprocess_data(anmeldungen):
    anmeldungen = anmeldungen[anmeldungen.Vorname != '']
    anmeldungen.columns = ["date", "first name", "last name", "nickname", "S", "email_unused", "from", "F", "T", "E", "email", "unnamed"]
    anmeldungen = anmeldungen.sort_values("S", ascending=False).reset_index(drop=True)
    anmeldungen["E"] = anmeldungen["E"].astype(np.ubyte)
    anmeldungen["T"] = anmeldungen["T"].astype(np.ubyte)
    anmeldungen["F"] = anmeldungen["F"].astype(np.ubyte)
    return anmeldungen


def import_data_from_url(url):
    anmeldungen = pd.read_csv(url)
    return _preprocess_data(anmeldungen)

def import_data_from_gdrive(spreadsheet_id, worksheet="Formularantworten 1", service_account_file="service_account.json"):
    gc = gspread.service_account(filename=service_account_file) # type: ignore
    spreadsheet = gc.open_by_key(spreadsheet_id)
    anmeldungen = pd.DataFrame(spreadsheet.worksheet(worksheet).get_all_records())
    return _preprocess_data(anmeldungen)
        

def _evaluate_assignment(no_teams, player_props, assignment):
    team_means = []
    team_vars = []
    for i in range(no_teams):
        team_props = player_props[assignment == i, :]
        
        means = np.mean(team_props[:,1:4], axis = 0)
        
        vrs = np.var(team_props[:, 1:4], axis = 0)
        
        team_means.append(means)
        
        team_vars.append(np.mean(vrs))
        #sds = np.std(player_props[:,1:4], axis = 0)
        
    outer_var = np.sum(np.var(team_means, axis=0))
    overall_outer_var = np.var(np.sum(team_means, axis=1))
    #avg_var = np.var(np.sum(team_means, axis=1))
    return overall_outer_var, outer_var, np.min(team_vars), np.max(team_vars)

def _score_fun(ev):
    return 2*ev[0] + ev[1] + 0.005 * ev[3]

def compute_assignments(anmeldungen, no_teams, max_tries=10000, no_cores=6):
    sex = anmeldungen["S"].replace(["Male matching", "Female matching"], ["M", "F"])
    experience = anmeldungen["E"]
    throwing = anmeldungen["T"]
    fitness = anmeldungen["F"]

    player_props = np.array([sex, experience, throwing, fitness]).T

    males = np.sum(player_props[:,0] == "M")
    females = np.sum(player_props[:,0] == "F")

    min_males = np.floor(males / no_teams).astype(int)
    min_females = np.floor(females / no_teams).astype(int)

    overhead_males = males - min_males * no_teams
    overhead_females = females - min_females * no_teams

    male_sorted = []
    female_sorted = []

    for i in range(no_teams):
        if i < overhead_males:
            male_sorted = male_sorted + [i] * (min_males + 1)
            
            female_sorted = female_sorted + [i] * min_females
        else:
            male_sorted = male_sorted + [i] * min_males
            if i < overhead_males + overhead_females:
                female_sorted = female_sorted + [i] * (min_females + 1)
            else:
                female_sorted = female_sorted + [i] * min_females
                
    male_sorted = [n for team, length in enumerate([min_males + 1] * overhead_males + [min_males] * (no_teams-overhead_males)) for n in [team] * length]
    female_sorted = [n for team, length in enumerate([min_females] * (no_teams-overhead_females) + [min_females + 1] * overhead_females) for n in [team] * length]

    def generate_and_evaluate_assignment(seed=1):
        rng = np.random.default_rng(seed)
        assignment = np.hstack(([0], rng.permutation(male_sorted[1:]), rng.permutation(female_sorted))).astype(np.ubyte)
        ev = _evaluate_assignment(no_teams, player_props, assignment)
        return assignment, ev
    
    assignments = Parallel(n_jobs=no_cores)(delayed(generate_and_evaluate_assignment)(i) for i in range(max_tries))

    scores = [_score_fun(r[1]) for r in assignments] # type: ignore

    min_score = np.min(scores)
    good_assignments = [(assg[0], score, assg[1]) for assg, score in zip(assignments, scores)] # type: ignore

    return sorted(good_assignments, key=lambda a : a[1])

def print_best_assignments(anmeldungen, assignments, no_suggestions = 5):
    anmeldungen = anmeldungen.copy()
    print("\nVar(TotalSkill) | Sum_Skill( Var(Skill) ) | Min/Max Var(Skill) within each team", "\n\nE = Experience", "\nT = Throwing", "\nF = Fitness\n\n")
    for i in range(no_suggestions):
        asg = assignments[i]
        
        score = asg[1]
        
        print("Vorschlag %d (score = %.4f)" % (i, score), 
            "\n---------------------------------",
            "\n%.4f | %.4f | %.4f | %.4f" % asg[2],
            "\n---------------------------------", "\n")
        anmeldungen["Team"] = asg[0] + 1

        means = anmeldungen.groupby("Team")[["E", "T", "F"]].mean()
        vars = anmeldungen.groupby("Team")[["E", "T", "F"]].var()

        print("Team  | E       |  T      | F")
        for k in range(len(means)):
            m = means.iloc[k]
            v = vars.iloc[k]
            print("%d     | %.1f (%1.f) | %.1f (%1.f) | %.1f (%1.f)" % (k+1, m["E"], v["E"], m["T"], v["T"], m["F"], v["F"]))

        print("Range | %.1f     | %.1f     | %.1f " % (np.ptp(means["E"]), np.ptp(means["T"]), np.ptp(means["F"])))
        print("---------------------------------")
        
        for team, players in anmeldungen.groupby("Team"):
            print("Team %d" % (team), "\n------")
            print("Mean E: %.1f, T: %.1f, F: %.1f" % (np.mean(players["E"]), np.mean(players["T"]), np.mean(players["F"])), "\n------")
            print(players[["first name", "last name", "S", "from", "E", "T", "F"]].reset_index(drop=True)) #"from",
            print("\n")