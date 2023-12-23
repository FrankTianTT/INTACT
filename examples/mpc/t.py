import hydra


@hydra.main(version_base="1.1", config_path="conf", config_name="main")
def main(cfg):
    print(cfg)
    print(cfg.env_name)


if __name__ == '__main__':
    main()
