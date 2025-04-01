import torch
import torch.nn.functional as F
import torch.nn as nn

# --- 1. Attribute Loss (Lattr) ---
def compute_attribute_loss(f_attr: torch.Tensor, f_attr_prime: torch.Tensor) -> torch.Tensor:
    """
    Calculates the L2 loss between original and reconstructed attribute features.
    Lattr = ||f_attr - f'_attr||^2 / 2 (as in paper's Lattr definition potentially,
                                        or just MSE)
    We'll use standard MSE loss provided by PyTorch for simplicity and stability.

    Args:
        f_attr (torch.Tensor): Attribute features from the original image Vt.
                               Shape: (B, C, H, W) or similar.
        f_attr_prime (torch.Tensor): Attribute features from the reconstructed image V't.
                                     Shape: (B, C, H, W) or similar.

    Returns:
        torch.Tensor: The calculated attribute loss (scalar).
    """
    loss = F.mse_loss(f_attr_prime, f_attr) # PyTorch MSE averages over all elements
    return loss

# --- 2. Reconstruction Loss (Lrec) ---
def compute_reconstruction_loss(vt: torch.Tensor,
                                vt_prime: torch.Tensor,
                                is_same_identity: torch.Tensor,
                                loss_type: str = 'l1') -> torch.Tensor:
    """
    Calculates the reconstruction loss (L1 or L2) between Vt and V't,
    only when the identity used for reconstruction (frid) is the same as
    the original identity (fgid).

    Args:
        vt (torch.Tensor): The original target tensor (e.g., VAE latent of Vt).
                           Shape: (B, C, H, W).
        vt_prime (torch.Tensor): The reconstructed tensor V't from the Decoder.
                                 Shape: (B, C, H, W).
        is_same_identity (torch.Tensor): A boolean tensor of shape (B,) or (B, 1)
                                         indicating for each sample in the batch
                                         if frid == fgid. True if same, False otherwise.
        loss_type (str): Type of loss to use ('l1' or 'l2'). Default is 'l1'.

    Returns:
        torch.Tensor: The calculated reconstruction loss (scalar). Averages over the batch dimension
                      for samples where is_same_identity is True.
    """
    # Ensure is_same_identity can be broadcasted for masking
    # Reshape to (B, 1, 1, 1) to match vt/vt_prime dimensions for element-wise multiplication
    mask = is_same_identity.view(-1, 1, 1, 1).float() # Convert boolean to float (1.0 or 0.0)

    # Calculate the chosen loss only for masked elements
    if loss_type == 'l1':
        # Calculate L1 loss element-wise
        loss_elementwise = F.l1_loss(vt_prime, vt, reduction='none')
    elif loss_type == 'l2':
        # Calculate L2 loss element-wise
        loss_elementwise = F.mse_loss(vt_prime, vt, reduction='none')
    else:
        raise ValueError("loss_type must be 'l1' or 'l2'")

    # Apply the mask
    masked_loss_elementwise = loss_elementwise * mask

    # Calculate the mean loss only over the samples where the mask is 1
    # Sum the loss for the masked samples and divide by the number of masked samples
    # Add a small epsilon to avoid division by zero if no samples have the same identity
    num_valid_samples = torch.sum(is_same_identity).float().clamp(min=1e-6)
    loss = torch.sum(masked_loss_elementwise) / num_valid_samples

    # Handle case where num_valid_samples is 0 (though clamped, good practice)
    if num_valid_samples < 1e-5:
         loss = torch.tensor(0.0, device=vt.device, dtype=vt.dtype) # Return 0 if no samples apply


    return loss


# --- 3. Triplet Identity Loss (Ltid) ---
def compute_triplet_identity_loss(fgid: torch.Tensor,
                                  f_gid_prime: torch.Tensor,
                                  frid: torch.Tensor,
                                  is_same_identity: torch.Tensor,
                                  margin: float = 0.5) -> torch.Tensor:
    """
    Calculates the Triplet Margin Identity Loss (Eq 5, adjusted for PyTorch).
    Ltid = max(cos(fgid, f'gid) - cos(fgid, frid) + margin, 0)
    Computed only when is_same_identity is False (i.e., frid != fgid).

    Args:
        fgid (torch.Tensor): Identity features from the original Vt. Shape: (B, EmbedDim).
        f_gid_prime (torch.Tensor): Identity features from the reconstructed V't. Shape: (B, EmbedDim).
        frid (torch.Tensor): Identity features used for reconstruction. Shape: (B, EmbedDim).
        is_same_identity (torch.Tensor): Boolean tensor shape (B,) or (B, 1). True if frid == fgid.
        margin (float): The margin for the triplet loss.

    Returns:
        torch.Tensor: The calculated triplet identity loss (scalar). Averages over the batch dimension
                      for samples where is_same_identity is False.
    """
    # Ensure inputs are normalized (though FaceRecognizer should already do this)
    fgid = F.normalize(fgid, p=2, dim=1)
    f_gid_prime = F.normalize(f_gid_prime, p=2, dim=1)
    frid = F.normalize(frid, p=2, dim=1)

    # Calculate cosine similarities
    # F.cosine_similarity computes similarity between corresponding elements in batches
    cos_orig_recon = F.cosine_similarity(fgid, f_gid_prime, dim=1) # Shape (B,)
    cos_orig_input = F.cosine_similarity(fgid, frid, dim=1)        # Shape (B,)

    # Calculate the triplet loss term per sample
    loss_per_sample = F.relu(cos_orig_recon - cos_orig_input + margin) # relu implements max(0, x)

    # Create mask for samples where frid != fgid
    mask = (~is_same_identity.squeeze()).float() # Negate and convert boolean to float (1.0 or 0.0)

    # Apply the mask
    masked_loss_per_sample = loss_per_sample * mask

    # Average the loss only over the samples where the mask is 1
    num_valid_samples = torch.sum(mask).float().clamp(min=1e-6)
    loss = torch.sum(masked_loss_per_sample) / num_valid_samples

    if num_valid_samples < 1e-5:
         loss = torch.tensor(0.0, device=fgid.device, dtype=fgid.dtype)

    return loss


# --- 4. Adversarial Loss (Ladv) ---
# We need separate functions for Generator and Discriminator losses

# 4.1 Generator Adversarial Loss
def compute_G_adv_loss(fake_scores: torch.Tensor, loss_type: str = 'bce') -> torch.Tensor:
    """
    Calculates the adversarial loss for the Generator.
    The goal is to make the discriminator classify fake samples as real.

    Args:
        fake_scores (torch.Tensor): Output scores from the Discriminator for generated samples (V't).
                                    Shape depends on Discriminator (e.g., (B, 1, H', W') for PatchGAN).
        loss_type (str): Type of GAN loss ('bce', 'hinge', etc.). Default 'bce'.

    Returns:
        torch.Tensor: The calculated generator adversarial loss (scalar).
    """
    # Target labels for fake samples should be "real" (1.0) for the generator
    target_real = torch.ones_like(fake_scores)

    if loss_type == 'bce':
        # Use Binary Cross Entropy with Logits
        loss = F.binary_cross_entropy_with_logits(fake_scores, target_real)
    elif loss_type == 'hinge':
         # Hinge loss for generator: -D(G(z))
         loss = -torch.mean(fake_scores)
    # Add other GAN loss variants if needed (e.g., WGAN-GP, LSGAN)
    else:
        raise ValueError("Unsupported GAN loss_type for Generator")

    return loss

# 4.2 Discriminator Adversarial Loss
def compute_D_adv_loss(real_scores: torch.Tensor,
                         fake_scores: torch.Tensor,
                         loss_type: str = 'bce') -> torch.Tensor:
    """
    Calculates the adversarial loss for the Discriminator.
    The goal is to correctly classify real samples as real and fake samples as fake.

    Args:
        real_scores (torch.Tensor): Output scores from the Discriminator for real samples (Vt).
        fake_scores (torch.Tensor): Output scores from the Discriminator for generated samples (V't).
        loss_type (str): Type of GAN loss ('bce', 'hinge', etc.). Default 'bce'.

    Returns:
        torch.Tensor: The calculated discriminator adversarial loss (scalar).
    """
    # Target labels for real samples are 1.0, for fake samples are 0.0
    target_real = torch.ones_like(real_scores)
    target_fake = torch.zeros_like(fake_scores)

    if loss_type == 'bce':
        # BCE loss for real samples
        loss_real = F.binary_cross_entropy_with_logits(real_scores, target_real)
        # BCE loss for fake samples
        loss_fake = F.binary_cross_entropy_with_logits(fake_scores, target_fake)
        # Total discriminator loss is the average of the two
        loss = (loss_real + loss_fake) * 0.5
    elif loss_type == 'hinge':
        # Hinge loss for discriminator: max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
        loss_real = torch.mean(F.relu(1.0 - real_scores))
        loss_fake = torch.mean(F.relu(1.0 + fake_scores))
        loss = (loss_real + loss_fake) * 0.5
    # Add other GAN loss variants if needed
    else:
        raise ValueError("Unsupported GAN loss_type for Discriminator")

    return loss


# --- 5. Total FAL Loss for Generator ---
def compute_G_total_loss(l_g_adv: torch.Tensor,
                         l_attr: torch.Tensor,
                         l_rec: torch.Tensor,
                         l_tid: torch.Tensor,
                         lambda_adv: float = 1.0,
                         lambda_attr: float = 10.0,
                         lambda_rec: float = 10.0,
                         lambda_tid: float = 1.0) -> torch.Tensor:
    """
    Calculates the total loss for the Generator (Eattr + Dec) by combining
    individual losses with their respective weights.

    Args:
        l_g_adv (torch.Tensor): Generator adversarial loss.
        l_attr (torch.Tensor): Attribute loss.
        l_rec (torch.Tensor): Reconstruction loss.
        l_tid (torch.Tensor): Triplet identity loss.
        lambda_adv (float): Weight for adversarial loss.
        lambda_attr (float): Weight for attribute loss (default from paper: 10).
        lambda_rec (float): Weight for reconstruction loss (default from paper: 10).
        lambda_tid (float): Weight for triplet identity loss (default from paper: 1).

    Returns:
        torch.Tensor: The total weighted loss for the generator (scalar).
    """
    total_loss = (lambda_adv * l_g_adv +
                  lambda_attr * l_attr +
                  lambda_rec * l_rec +
                  lambda_tid * l_tid)
    # You might want to return individual weighted losses as well for logging
    # return total_loss, lambda_adv * l_g_adv, ...
    return total_loss


# --- Optional: Example Usage/Testing ---
if __name__ == '__main__':
    print("--- Testing Loss Functions ---")
    B = 4 # Batch size
    EmbedDim = 512
    C, H, W = 4, 80, 80
    Hp, Wp = 9, 9 # PatchGAN output size

    # --- Dummy Data ---
    # Features
    f_attr = torch.randn(B, 1280, 20, 20)
    f_attr_prime = torch.randn(B, 1280, 20, 20)
    # Latents/Images
    vt = torch.randn(B, C, H, W)
    vt_prime = torch.randn(B, C, H, W)
    # IDs
    fgid = F.normalize(torch.randn(B, EmbedDim))
    frid_same = fgid.clone() # Case 1: frid == fgid
    frid_diff = F.normalize(torch.randn(B, EmbedDim)) # Case 2: frid != fgid
    f_gid_prime_from_same = F.normalize(torch.randn(B, EmbedDim)) # Fake reconstruction ID when frid == fgid
    f_gid_prime_from_diff = F.normalize(torch.randn(B, EmbedDim)) # Fake reconstruction ID when frid != fgid
    # Discriminator scores
    real_scores = torch.randn(B, 1, Hp, Wp)
    fake_scores = torch.randn(B, 1, Hp, Wp)
    # Identity mask (example: half same, half different)
    is_same_identity = torch.tensor([True, True, False, False])

    # --- Test Calculations ---
    margin = 0.5
    lambda_adv, lambda_attr, lambda_rec, lambda_tid = 1.0, 10.0, 10.0, 1.0

    # Test Lattr
    l_attr = compute_attribute_loss(f_attr, f_attr_prime)
    print(f"Lattr: {l_attr.item():.4f}")

    # Test Lrec (using frid_same, should compute loss for first 2 samples)
    l_rec_same = compute_reconstruction_loss(vt, vt_prime, is_same_identity, loss_type='l1')
    print(f"Lrec (where same): {l_rec_same.item():.4f}")
    # Test Lrec (using frid_diff, should compute loss for last 2 samples - wait, Lrec only uses is_same_id)
    l_rec_test = compute_reconstruction_loss(vt, vt_prime, torch.tensor([False, False, False, False]), loss_type='l1')
    print(f"Lrec (all different, should be ~0): {l_rec_test.item():.4f}")


    # Test Ltid (using frid_diff, should compute loss for last 2 samples)
    l_tid_diff = compute_triplet_identity_loss(fgid, f_gid_prime_from_diff, frid_diff, is_same_identity, margin=margin)
    print(f"Ltid (where different): {l_tid_diff.item():.4f}")
     # Test Ltid (using frid_same, should compute loss for first 2 samples - wait, Ltid only uses is_same_id=False)
    l_tid_test = compute_triplet_identity_loss(fgid, f_gid_prime_from_same, frid_same, is_same_identity, margin=margin)
    print(f"Ltid (check where same, should be ~0): {l_tid_test.item():.4f}") # Should be 0 because mask applies only where ~is_same


    # Test Ladv (BCE)
    l_g_adv_bce = compute_G_adv_loss(fake_scores, loss_type='bce')
    l_d_adv_bce = compute_D_adv_loss(real_scores, fake_scores, loss_type='bce')
    print(f"L_G_adv (BCE): {l_g_adv_bce.item():.4f}")
    print(f"L_D_adv (BCE): {l_d_adv_bce.item():.4f}")

    # Test Ladv (Hinge)
    l_g_adv_hinge = compute_G_adv_loss(fake_scores, loss_type='hinge')
    l_d_adv_hinge = compute_D_adv_loss(real_scores, fake_scores, loss_type='hinge')
    print(f"L_G_adv (Hinge): {l_g_adv_hinge.item():.4f}")
    print(f"L_D_adv (Hinge): {l_d_adv_hinge.item():.4f}")

    # Test Total G Loss
    # Use Lrec and Ltid calculated based on the mixed is_same_identity mask
    total_g_loss = compute_G_total_loss(l_g_adv_bce, l_attr, l_rec_same, l_tid_diff,
                                        lambda_adv, lambda_attr, lambda_rec, lambda_tid)
    print(f"\nTotal G Loss (Example): {total_g_loss.item():.4f}")

    print("\n--- Loss Function Tests Finished ---")