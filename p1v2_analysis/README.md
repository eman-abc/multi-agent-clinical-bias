# p1v2_analysis: behavioural study, rebuilt run
Put this folder inside `C:\Mitacs Research\multi-agent-clinical-bias` (next to `p1v2_data` and `results`). Use the `mech` environment (numpy, scipy, matplotlib).
```powershell
cd "C:\Mitacs Research\multi-agent-clinical-bias"
C:\venvs\mech\Scripts\Activate.ps1
python p1v2_analysis\tests\test_mock.py                       # must end with MOCK TEST OK
Copy-Item p1v2_analysis\P1V2_ANALYSIS_PLAN.md p1v2_analysis\PLAN_p1v2.md
notepad p1v2_analysis\PLAN_p1v2.md                            # fill the 10 probabilities (your own beliefs)
python p1v2_analysis\freeze_p1.py freeze --plan p1v2_analysis\PLAN_p1v2.md --name p1v2
python p1v2_analysis\freeze_p1.py verify p1v2_analysis\lock_p1v2.json     # expect FROZEN-OK
python p1v2_analysis\analyze_p1v2.py --lock p1v2_analysis\lock_p1v2.json  # about 1-3 minutes
notepad p1v2_analysis\results\P1V2_LEDGER.md
```
Outputs in `p1v2_analysis\results\`: CSV tables, `P1V2_LEDGER.md`, figures `fig_p1_*.png/pdf`.
Freeze only after the run has finished; editing any answer file afterwards makes verification fail.
