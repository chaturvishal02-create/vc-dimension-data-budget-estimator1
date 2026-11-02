import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import get_dataset
from src.preprocessing import split_and_preprocess
from src.evaluation import run_full_experiment, build_comparison_dataframe
from src.vc_dimension import estimate_vc_dimension
from src.data_budget import estimate_data_budget
from src.report_generator import generate_pdf_report, generate_excel_report

print("--- Running Platform Smoke Test ---")
ds = get_dataset('Iris')
X_tr, X_te, y_tr, y_te, prep = split_and_preprocess(ds['X'], ds['y'], task_type="classification")
print(f"Dataset loaded: {ds['dataset_name']} ({ds['n_samples']} samples, {ds['n_features']} features)")

models_to_test = ["Perceptron", "Decision Tree", "Random Forest", "Support Vector Machine"]
results = []

for m in models_to_test:
    r = run_full_experiment(
        m, X_tr, y_tr, X_te, y_te,
        epsilon=0.10, delta=0.05, n_features=ds['n_features'],
        task_type="classification",
    )
    results.append(r)
    vc = r['vc_info']['vc_dim']
    pac = r['pac_bound']['m']
    emp = r['empirical']['sample_count']
    print(f"Model: {m:<25} | VC Dim: {vc:<5} | PAC: {pac:<6} | Emp: {str(emp):<5} | Val Acc: {r['final_val_acc']*100:.1f}%")

df_comp = build_comparison_dataframe(results)
print(f"Comparison dataframe generated: {len(df_comp)} rows")

# Test Data Budget Estimator
b = estimate_data_budget(
    theoretical_m=results[0]['pac_bound']['m'],
    empirical_achieved=results[0]['empirical']['achieved'],
    empirical_m=results[0]['empirical']['sample_count'],
    current_dataset_size=len(X_tr),
    lc_result=results[0]['learning_curve'],
    target_error=0.10,
    cost_per_sample=1.50,
)
print(f"Budget Estimator Conservatism Ratio: {b['conservatism_ratio']}x")

# Test PDF and Excel Generation
pdf_data = generate_pdf_report(results, "Iris", ds, epsilon=0.10, delta=0.05)
print(f"PDF Report generated: {len(pdf_data):,} bytes")

excel_data = generate_excel_report(results, "Iris", ds, epsilon=0.10, delta=0.05)
print(f"Excel Report generated: {len(excel_data):,} bytes")

print("--- ALL SMOKE TESTS COMPLETED SUCCESSFULLY ---")
