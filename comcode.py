import argparse


def main():
    parser = argparse.ArgumentParser(description='Process EDF file with selected model.')
    parser.add_argument('model', type=str, help='Select the model: detspike, seizure, sleep, Report_generation, ICA, CBA')
    parser.add_argument('edf_path', type=str, help='Path to the EDF file')
    args = parser.parse_args()

    model_name = args.model.strip()
    edf_path = args.edf_path

    model_key = model_name.lower()

    try:
        if model_key == 'detspike':
            from spikeNet import detspike
            output_text = detspike(edf_path)
            print("{%s}" % output_text)
        elif model_key == 'seizure':
            from seizure import supervised
            output_text = supervised(edf_path)
            print(output_text)
        elif model_key == 'sleep':
            from sleep import sleep
            output_text = sleep(edf_path)
            print("{%s}" % output_text)
        elif model_key in ('report_generation', 'report'):
            from report import Report_generation
            report = Report_generation(edf_path, ascii_safe=True)
            print("{%s}" % report)
        elif model_key == 'ica':
            from ICA import ICAArtifactRemover
            output_text = ICAArtifactRemover().fit_transform(edf_path)
            print("{%s}" % output_text)
        elif model_key == 'cba':
            from Cbra import Cbramod_run
            output_text = Cbramod_run(edf_path)
            print("{%s}" % output_text)
        elif model_key in ('rsn', 'supervised_rsn'):
            from RSN_seizure import supervised_rsn
            output_text = supervised_rsn(edf_path)
            print("{%s}" % output_text)
        else:
            print("Invalid model name. Please choose from detspike, seizure, sleep, Report_generation, ICA, CBA.")
    except Exception as e:
        print(f"{{运行失败: {e}}}")
        raise


if __name__ == "__main__":
    main()
