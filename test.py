from util import get_device, TrainMode, load_model, validate_one_epoch, get_config_name, TrainMethod

import os
import wandb
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    print('\nLogin to weights and biases ...')
    wandb_key = os.getenv('WANDB_KEY')
    wandb.login(key=wandb_key)

    hydra_cfg = HydraConfig.get()
    run = wandb.init(
        project=cfg.wandb.project,
        name= 'test_' + get_config_name(hydra_cfg.job.config_name),
    )
    
    device = get_device()
    print(f'{device} is used for testing...')

    print('\nLoading datasets ...')
    # with poor image qualities
    transform_test = hydra.utils.instantiate(cfg.transform.test)
    transform_target = hydra.utils.instantiate(cfg.transform.target)

    dataset_test = hydra.utils.instantiate(
        cfg.dataset,
        type=TrainMode.TEST.value,
        transform_data=transform_test,
        transform_target=transform_target,
    )
    dataloader_test = hydra.utils.instantiate(cfg.dataloader.test, dataset=dataset_test)

    # without poor image qualities
    hydra_cfg = HydraConfig.get()
    if hydra_cfg.runtime.choices.dataset == 'camus':
        dataset_test_noPoor = hydra.utils.instantiate(
            cfg.dataset,
            type=TrainMode.TEST.value,
            transform_data=transform_test,
            transform_target=transform_target,
            no_poor = True,
        )
        dataloader_test_noPoor = hydra.utils.instantiate(cfg.dataloader.test, dataset=dataset_test_noPoor)

    print('\nLoading model ...')
    model = hydra.utils.instantiate(cfg.trainer.model)
    

    if cfg.trainer.mode == TrainMethod.LORA.value:
        pretrained_model = hydra.utils.instantiate(cfg.trainer.pretrained_model)
        model = model(pretrained_model)

    if not cfg.model_id:
        print('Attribut model_id with wandb run id not set in config file')
        return

    model = load_model(model, cfg.model_id, project=cfg.wandb.project)
    model = model.to(device)

    criterion = hydra.utils.instantiate(cfg.trainer.criterion)
    metric = hydra.utils.instantiate(cfg.trainer.metric)
    metric = metric(device)

    eval_metric = cfg.trainer.eval_metric

    print('\nTesting model ...')
    results_test = validate_one_epoch(1, model, dataloader_test, criterion, device, metric)

    if hydra_cfg.runtime.choices.dataset == 'camus':
        results_test_noPoor = validate_one_epoch(1, model, dataloader_test_noPoor, criterion, device, metric)

    print('\nResults:')
    if hydra_cfg.runtime.choices.dataset == 'camus':
        print("{} with poor image qualities:\t{:.3f}".format(eval_metric, results_test[eval_metric]))
        print("{} without poor image qualities:\t{:.3f}".format(eval_metric, results_test_noPoor[eval_metric]))

        run.log({f'test_{eval_metric}': results_test[eval_metric],
                 f'test_{eval_metric}_noPoor': results_test_noPoor[eval_metric]})
    else:
        print("{}:\t{:.3f}".format(eval_metric, results_test[eval_metric]))

        run.log({f'test_{eval_metric}': results_test[eval_metric]})

    run.finish()

if __name__ == '__main__':
    main()