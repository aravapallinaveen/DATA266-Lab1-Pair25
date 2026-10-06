import torch
from torch import nn
from config import CONFIG
MSE = nn.MSELoss()
L1 = nn.L1Loss()

def adversarial(prediction, real):
    return MSE(prediction, torch.ones_like(prediction) if real else torch.zeros_like(prediction))

def generator_losses(models, a, b):
    ga, gb = models["G_A2B"], models["G_B2A"]
    fake_b, fake_a = ga(a), gb(b)
    rec_a, rec_b = gb(fake_b), ga(fake_a)
    identity_b, identity_a = ga(b), gb(a)
    losses = dict(G_A2B_adv=adversarial(models["D_B"](fake_b), True),
                  G_B2A_adv=adversarial(models["D_A"](fake_a), True),
                  cycle_A=CONFIG["lambda_cycle"]*L1(rec_a, a),
                  cycle_B=CONFIG["lambda_cycle"]*L1(rec_b, b),
                  identity_A=CONFIG["lambda_identity"]*L1(identity_a, a),
                  identity_B=CONFIG["lambda_identity"]*L1(identity_b, b))
    losses["G_total"] = sum(losses.values())
    tensors = dict(real_A=a, real_B=b, fake_A=fake_a, fake_B=fake_b,
                   reconstructed_A=rec_a, reconstructed_B=rec_b,
                   identity_A=identity_a, identity_B=identity_b)
    return losses, tensors

def discriminator_loss(model, real, fake):
    return .5*(adversarial(model(real), True) + adversarial(model(fake.detach()), False))
