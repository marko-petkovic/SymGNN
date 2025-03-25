import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

import dotenv
from pathlib import Path

import numpy as np
import os

from symgraph.model.model import GNN
from symgraph.model.alignn import ALIGNN
from symgraph.model.model_utils import get_loading_and_derivative, langmuir_freundlich_2s, get_loading_weight
from symgraph.dataset import create_dataloaders
from symgraph.plot_utils import plot_isotherms
from symgraph.sys_utils import PROJECT_ROOT

from tqdm import tqdm

import wandb


import hydra
import omegaconf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

torch._dynamo.config.capture_scalar_outputs = True


def train_eval_epoch(model, loss_fn, trainloader, valloader, opt, lr_scheduler, device, epoch, mu, std, 
                     q_coef, n_steps=50, window_ratio=0.25, iso_noise=0.02, last_epoch_scheduler=100):
    
    wandb.log({'learning_rate': opt.param_groups[0]['lr'],
               'epoch': epoch})

    model.train()
    pbar = tqdm(trainloader, desc=f'Epoch {epoch}', ncols=0)

    train_loss = 0
    train_error = 0
    train_loss_h = 0
    train_loss_q = 0

    n_iso = 0

    p = torch.linspace(1, 7, n_steps).to(device).unsqueeze(-1)
    # dp = p[1] - p[0]
    # p_in = torch.cat([p, p[[-1]] + dp], dim=0) - dp/2
    # p_in = p

    window_size = max(1, int(window_ratio*n_steps))

    for data in pbar:
        opt.zero_grad()
        data = data.to(device)


        start_idx = np.random.randint(0, n_steps - window_size)

        p_in = p[start_idx:start_idx+window_size]

        # add noise to slightly perturb the isotherm parameters
        # this is to prevent the model from overfitting to the exact isotherm parameters
        iso_params = data.iso_params

        noise_params = (torch.rand_like(iso_params) * iso_noise) - iso_noise/2

        iso_params = iso_params * (1 + noise_params)


        # q = langmuir_freundlich_2s(iso_params, 10**p_in).T

        q, q_prime = get_loading_and_derivative(iso_params, p_in)

        # print("Loading dist")
        # print(q.min(), q.max(), q.mean(), q.shape, q.isnan().sum())
        

        out, q_prime_hat = model(data, pres=p_in)

        out = out.squeeze(-1)

        loss_h = loss_fn(out, data.y)
        
        q_prime_hat = q_prime_hat[data.iso]
        q_prime = q_prime[data.iso]
        
        n_iso_b = data.iso.sum().item()
        n_iso += n_iso_b

        loss_q = (q_prime_hat - q_prime).pow(2).mean(-1)
        
        loss_q = loss_q.mean() if data.iso.sum() > 0 else torch.tensor(0.0, device=device)

        loss = loss_h + q_coef*loss_q

        loss.backward()

        clip_grad_norm_(model.parameters(), 1.0)

        opt.step()

        train_loss += loss.item()
        train_loss_h += loss_h.item()
        train_loss_q += loss_q.item() * n_iso_b

        with torch.no_grad():
            error = torch.abs(out - data.y)
            error = error * std
            error = error.mean()
            train_error += error.item()
        
        wandb.log({'train_loss_step': loss.item(), 
                   'error_step': error.item(),
                   'train_loss_h_step': loss_h.item(),
                   'train_loss_q_step': loss_q.item(),})

        pbar.set_postfix({'loss': f'{loss.item():.2f}'})

    train_loss /= len(trainloader)
    pbar.set_postfix({'loss': f'{train_loss:.3f}'})

    wandb.log({'train_loss_epoch': train_loss, 
               'train_loss_h_epoch': train_loss_h/len(trainloader),
               'train_loss_q_epoch': train_loss_q/n_iso,
               'error_epoch': train_error/len(trainloader),
               'epoch': epoch})

    model.eval()

    val_loss = 0
    val_error = 0
    val_loss_h = 0
    val_loss_q = 0

    n_iso = 0

    # with torch.no_grad():

    true_iso = []
    pred_iso = []

    # p_in = p
    dp = p[1] - p[0]
    p_in = torch.cat([p, p[[-1]] + dp], dim=0) - dp/2

    # p_in_true = 10**p_in
    # p_eval_true = (p_in_true[1:] + p_in_true[:-1])/2

    # p_in = p


    for data in valloader:

        data = data.to(device)
        q = langmuir_freundlich_2s(data.iso_params, 10**p).T
        
        with torch.no_grad():
            out, q_prime_hat = model(data, pres=p_in)
            out = out.squeeze(-1)

        q_hat = torch.cumulative_trapezoid(q_prime_hat, p_in.squeeze(-1), dim=-1)

        q = q[data.iso]
        q_hat = q_hat[data.iso]

        n_iso_b = data.iso.sum().item()
        n_iso += n_iso_b
        
        if data.iso.sum() > 0:
            true_iso.extend(q.cpu().numpy().tolist())
            pred_iso.extend(q_hat.cpu().numpy().tolist())


        loss_h = loss_fn(out, data.y)
        loss_q = (q_hat - q).pow(2).mean(-1)

        # loss_q = loss_q * (data.iso*1.0)
        loss_q = loss_q.mean() if data.iso.sum() > 0 else torch.tensor(0.0, device=device)

        loss = loss_h + q_coef*loss_q

        val_loss += loss.item()
        val_loss_h += loss_h.item()
        val_loss_q += loss_q.item() * n_iso_b

        error = torch.abs(out - data.y)
        error = error * std
        error = error.mean()

        val_error += error.item()
    
    val_loss /= len(valloader)
    val_loss_h /= len(valloader)
    val_loss_q /= n_iso
    val_error /= len(valloader)

    print(f'Epoch {epoch} Train Loss: {train_loss:.3f} Val Loss: {val_loss:.3f} Train Error: {train_error/len(trainloader):.3f} Val Error: {val_error:.3f}')


    if epoch <= last_epoch_scheduler:
        lr_scheduler.step()


    # select 10 random isotherms to plot
    idx = np.random.choice(len(true_iso), 10, replace=False)

    true_iso = np.array(true_iso)[idx]
    pred_iso = np.array(pred_iso)[idx]

    fig = plot_isotherms(true_iso, pred_iso, p.squeeze().detach().cpu().numpy())

    wandb.log({'isotherms': fig})

    wandb.log({'val_loss_epoch': val_loss,
                'val_error_epoch': val_error,
                'val_loss_h_epoch': val_loss_h,
                'val_loss_q_epoch': val_loss_q,
                'epoch': epoch})
    
    return train_loss, val_loss

@torch.no_grad()
def test(model, testloader, device, mu, std, metrics=[F.mse_loss, F.l1_loss], pres_points=100):

    model.eval()

    test_metrics = [0. for _ in metrics]
    test_metrics_iso = [0. for _ in metrics]
    in_ci = 0
    total = 0

    true = []
    pred = []
    err = []

    true_iso = []
    pred_iso = []
    total_iso = 0

    p = torch.linspace(1, 7, pres_points).to(device).unsqueeze(-1)
    dp = p[1] - p[0]
    p_in = torch.cat([p, p[[-1]] + dp], dim=0) - dp/2

    # p_in_true = 10**p_in
    # p_eval_true = (p_in_true[1:] + p_in_true[:-1])/2

    # p_in = p

    for data in testloader:
    
        data = data.to(device)
        
        # pres = pres.repeat(data.y.size(0), 1)

        out, q_prime_hat = model(data, pres=p_in)
        true_isotherm = langmuir_freundlich_2s(data.iso_params, 10**p).T
        pred_isotherm = torch.cumulative_trapezoid(q_prime_hat, p_in.squeeze(-1), dim=-1)
        iso_mask = data.iso == 1
        n_iso = iso_mask.sum().item()

        
        total_iso += n_iso

        true_isotherm = true_isotherm[iso_mask]
        pred_isotherm = pred_isotherm[iso_mask]

        if n_iso > 0:
            true_iso.extend(true_isotherm.cpu().numpy().tolist())
            pred_iso.extend(pred_isotherm.cpu().numpy().tolist())

        out = out.squeeze(-1)

        for i in range(len(metrics)):

            test_metrics[i] += metrics[i](out*std, data.y*std).item()
            if n_iso > 0:
                test_metrics_iso[i] += metrics[i](pred_isotherm, true_isotherm).item()*n_iso

        true.extend(data.y.cpu().numpy().tolist())
        pred.extend(out.cpu().numpy().tolist())

        ci_bool = (out >= data.y - data.hoa_err) & (out <= data.y + data.hoa_err)
        ci_int = ci_bool.sum().item()

        in_ci += ci_int
        total += len(ci_bool)
        # err.extend(data.hoa_err.cpu().numpy().tolist())

    test_metrics = [m/len(testloader) for m in test_metrics]
    test_metrics_iso = [m/total_iso for m in test_metrics_iso]


    for i, m in enumerate(test_metrics):
        wandb.log({f'test_{metrics[i].__name__}': m})

        print(f'Test {metrics[i].__name__}: {m:.4f}')
    
    acc = in_ci/total
    wandb.log({'accuracy (CI)': acc})

    for i, m in enumerate(test_metrics_iso):
        wandb.log({f'test_{metrics[i].__name__}_iso': m})

        print(f'Test {metrics[i].__name__} (Iso): {m:.4f}')

    true = np.array(true)
    pred = np.array(pred)

    true_iso = np.array(true_iso)
    pred_iso = np.array(pred_iso)

    return true, pred, true_iso, pred_iso





def run(cfg: DictConfig):

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    
    trainloader, valloader, testloader, mu, std = create_dataloaders(**cfg.dataset)

    if cfg.model.name == 'ALIGNN':
        model = ALIGNN(**cfg.model).to(cfg.device)
    else:
        model = GNN(**cfg.model).to(cfg.device)

    
    # model = torch.compile(model, dynamic=True, fullgraph=True)
    # except:
    #     print('---------------------------------')
    #     print('######## COMPILING FAILED #######')
    #     print('---------------------------------')
    #     pass

    opt = AdamW(model.parameters(), **cfg.optim.optimizer)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **cfg.optim.lr_scheduler)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, **cfg.optim.lr_scheduler)

    loss_fn = nn.MSELoss()

    wandb.init(project=cfg.logging.wandb.project, entity=cfg.logging.wandb.entity, name=cfg.expname, config=cfg_dict)
    wandb.watch(model, **cfg.logging.wandb_watch)

    best_val_loss = float('inf')

    os.makedirs(f'{PROJECT_ROOT}/models/{cfg.expname}', exist_ok=True)

    # save mu and std
    np.save(f'{PROJECT_ROOT}/models/{cfg.expname}/mu.npy', mu)
    np.save(f'{PROJECT_ROOT}/models/{cfg.expname}/std.npy', std)

    for epoch in range(cfg.epochs):

        q_coef = get_loading_weight(epoch, **cfg.optim.loading_weight)            
        train_loss, val_loss = train_eval_epoch(model, loss_fn, trainloader, valloader, opt, lr_scheduler, cfg.device, epoch, mu, std, q_coef, **cfg.isotherm)

        if epoch != 0 and epoch % 10 == 0 and val_loss < best_val_loss and epoch >= cfg.optim.loading_weight.total_epochs:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{PROJECT_ROOT}/models/{cfg.expname}/best.pth')

    torch.save(model.state_dict(), f'{PROJECT_ROOT}/models/{cfg.expname}/final.pth')

    true, pred, true_iso, pred_iso = test(model, testloader, cfg.device, mu, std, pres_points=cfg.isotherm.n_steps)

    wandb.finish()

    # save true and pred
    np.save(f'{PROJECT_ROOT}/models/{cfg.expname}/true.npy', true)
    np.save(f'{PROJECT_ROOT}/models/{cfg.expname}/pred.npy', pred)

    np.save(f'{PROJECT_ROOT}/models/{cfg.expname}/true_iso.npy', true_iso)
    np.save(f'{PROJECT_ROOT}/models/{cfg.expname}/pred_iso.npy', pred_iso)

    # save config
    with open(f'{PROJECT_ROOT}/models/{cfg.expname}/config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == '__main__':
    main()

