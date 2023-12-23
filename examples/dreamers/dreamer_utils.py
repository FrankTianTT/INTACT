import torch
import tensordict


def recover_pixels(pixels, stats):
    return (
        (255 * (pixels * stats["scale"] + stats["loc"]))
        .clamp(min=0, max=255)
        .to(torch.uint8)
    )


@torch.inference_mode()
def call_record(
        logger,
        record,
        collected_frames,
        sampled_tensordict,
        stats,
        model_based_env,
        actor_model,
        cfg,
):
    # td_record = record(None)
    td_record = None
    if td_record is not None and logger is not None:
        for key, value in td_record.items():
            if key in ["r_evaluation", "total_r_evaluation"]:
                logger.log_scalar(
                    key,
                    value.detach().item(),
                    step=collected_frames,
                )
    # Compute observation reco
    if cfg.record_video and record._count % cfg.record_interval == 0:
        world_model_td = sampled_tensordict

        true_pixels = recover_pixels(world_model_td[("next", "pixels")], stats)

        reco_pixels = recover_pixels(world_model_td["next", "reco_pixels"], stats)

        world_model_td = world_model_td.select("state", "belief", "reward")
        world_model_td = model_based_env.rollout(
            max_steps=true_pixels.shape[1],
            policy=actor_model,
            auto_reset=False,
            tensordict=world_model_td[:, 0],
        )

        imagine_pxls = recover_pixels(
            model_based_env.decode_obs(world_model_td)["next", "reco_pixels"],
            stats,
        )

        stacked_pixels = torch.cat([true_pixels, reco_pixels, imagine_pxls], dim=-1)
        if logger is not None:
            logger.log_video(
                "pixels_rec_and_imag",
                stacked_pixels.detach().cpu(),
            )


def grad_norm(optimizer: torch.optim.Optimizer):
    sum_of_sq = 0.0
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            sum_of_sq += p.grad.pow(2).sum()
    return sum_of_sq.sqrt().detach().item()


def retrieve_stats_from_state_dict(obs_norm_state_dict):
    return {
        "loc": obs_norm_state_dict["loc"],
        "scale": obs_norm_state_dict["scale"],
    }


def match_length(batch_td: tensordict.TensorDict, length):
    assert len(batch_td.shape) == 2, "batch_td must be 2D"

    batch_size, seq_len = batch_td.shape
    # min multiple of length that larger than or equal to seq_len
    new_seq_len = (seq_len + length - 1) // length * length

    # pad with zeros
    matched_td = torch.stack(
        [tensordict.pad(td, [0, new_seq_len - seq_len]) for td in batch_td], 0
    ).contiguous()
    return matched_td
