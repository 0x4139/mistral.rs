use std::{io::Cursor, sync::Arc};

use base64::{engine::general_purpose::STANDARD, Engine};
use candle_core::{Device, Tensor};
use image::DynamicImage;
use uuid::Uuid;

use crate::{
    sequence::{Sequence, SequenceState, StopReason},
    ImageChoice, ImageGenerationResponse, ImageGenerationResponseFormat,
};

pub async fn send_image_responses(
    input_seqs: &mut [&mut Sequence],
    images: Vec<DynamicImage>,
) -> candle_core::Result<()> {
    if input_seqs.len() != images.len() {
        candle_core::bail!(
            "Input seqs len ({}) does not match images generated len ({})",
            input_seqs.len(),
            images.len()
        );
    }

    for (seq, image) in input_seqs.iter_mut().zip(images) {
        let choice = match seq
            .image_gen_response_format()
            .unwrap_or(ImageGenerationResponseFormat::Url)
        {
            ImageGenerationResponseFormat::Url => {
                let saved_file = match seq.image_gen_save_file() {
                    Some(path) => path.to_string_lossy().into_owned(),
                    None => format!("image-generation-{}.png", Uuid::new_v4()),
                };
                image
                    .save_with_format(&saved_file, image::ImageFormat::Png)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                ImageChoice {
                    url: Some(saved_file),
                    b64_json: None,
                }
            }
            ImageGenerationResponseFormat::B64Json => {
                let mut buffer = Vec::new();
                image
                    .write_to(&mut Cursor::new(&mut buffer), image::ImageFormat::Png)
                    .expect("Failed to encode image");
                let encoded = STANDARD.encode(&buffer);
                let serialized_b64 = format!("data:image/png;base64,{encoded}");
                ImageChoice {
                    url: None,
                    b64_json: Some(serialized_b64),
                }
            }
        };
        seq.add_image_choice_to_group(choice);

        let group = seq.get_mut_group();
        group
            .maybe_send_image_gen_response(
                ImageGenerationResponse {
                    created: seq.creation_time() as u128,
                    data: group.get_image_choices().to_vec(),
                },
                seq.responder(),
            )
            .await
            .map_err(candle_core::Error::msg)?;

        seq.set_state(SequenceState::Done(StopReason::GeneratedImage));
    }

    Ok(())
}

pub async fn send_speech_responses(
    input_seqs: &mut [&mut Sequence],
    pcms: &[Arc<Vec<f32>>],
    rates: &[usize],
    channels: &[usize],
) -> candle_core::Result<()> {
    if input_seqs.len() != pcms.len() {
        candle_core::bail!(
            "Input seqs len ({}) does not match pcms generated len ({})",
            input_seqs.len(),
            pcms.len()
        );
    }

    for (seq, (pcm, (rate, channel))) in input_seqs
        .iter_mut()
        .zip(pcms.iter().zip(rates.iter().zip(channels)))
    {
        seq.add_speech_pcm_to_group(pcm.clone(), *rate, *channel);

        let group = seq.get_mut_group();
        group
            .maybe_send_speech_response(seq.responder())
            .await
            .map_err(candle_core::Error::msg)?;

        seq.set_state(SequenceState::Done(StopReason::GeneratedSpeech));
    }

    Ok(())
}

/// The last `max` rows across a sequence's raw chunks (rows are the second-to-last axis), still on
/// their device; `None` keeps every row.
pub(crate) fn keep_last_rows(
    chunks: Vec<Tensor>,
    max: Option<usize>,
) -> candle_core::Result<Vec<Tensor>> {
    let Some(mut left) = max else {
        return Ok(chunks);
    };
    let mut kept = Vec::new();
    for t in chunks.into_iter().rev() {
        if left == 0 {
            break;
        }
        let axis = t.rank().saturating_sub(2);
        let rows = t.dim(axis)?;
        if rows <= left {
            left -= rows;
            kept.push(t);
        } else {
            kept.push(t.narrow(axis, rows - left, left)?);
            left = 0;
        }
    }
    kept.reverse();
    Ok(kept)
}

pub async fn send_raw_responses(
    input_seqs: &mut [&mut Sequence],
    logits_chunks: Vec<Vec<Tensor>>,
) -> candle_core::Result<()> {
    if logits_chunks.len() != input_seqs.len() {
        candle_core::bail!(
            "raw responses for {} sequences, got {} chunk lists",
            input_seqs.len(),
            logits_chunks.len()
        );
    }
    for (seq, chunks) in input_seqs.iter_mut().zip(logits_chunks) {
        let seq: &mut Sequence = seq;
        let chunks = keep_last_rows(chunks, seq.max_raw_rows)?
            .into_iter()
            .map(|t| t.to_device(&Device::Cpu))
            .collect::<candle_core::Result<Vec<_>>>()?;
        seq.add_raw_choice_to_group(chunks);

        let group = seq.get_mut_group();
        group
            .maybe_send_raw_done_response(seq.responder())
            .await
            .map_err(candle_core::Error::msg)?;

        seq.set_state(SequenceState::Done(StopReason::Length(0)));
    }

    Ok(())
}

pub async fn send_embedding_responses(
    input_seqs: &mut [&mut Sequence],
    embedings: Vec<Vec<f32>>,
) -> candle_core::Result<()> {
    if embedings.len() != input_seqs.len() {
        candle_core::bail!("Number of embeddings must match number of sequences..");
    }

    for (seq, embeddings) in input_seqs.iter_mut().zip(embedings) {
        seq.add_embedding_choice_to_group(embeddings);

        let group = seq.get_mut_group();
        group
            .maybe_send_embedding_done_response(seq.responder())
            .await
            .map_err(candle_core::Error::msg)?;

        seq.set_state(SequenceState::Done(StopReason::Length(0)));
    }

    Ok(())
}

#[cfg(test)]
mod raw_rows_tests {
    use super::keep_last_rows;
    use candle_core::{Device, Tensor};

    fn rows(n: usize, start: f32) -> Tensor {
        let values: Vec<f32> = std::iter::successors(Some(start), |v| Some(v + 1.0))
            .take(n)
            .collect();
        Tensor::from_vec(values, (1, n, 1), &Device::Cpu).unwrap()
    }

    fn values(chunks: &[Tensor]) -> Vec<f32> {
        chunks
            .iter()
            .flat_map(|t| t.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            .collect()
    }

    #[test]
    fn keeps_every_row_without_a_cap() {
        let kept = keep_last_rows(vec![rows(3, 0.0), rows(2, 3.0)], None).unwrap();
        assert_eq!(values(&kept), [0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn keeps_the_last_rows_across_chunks() {
        let kept = keep_last_rows(vec![rows(3, 0.0), rows(2, 3.0)], Some(3)).unwrap();
        assert_eq!(values(&kept), [2.0, 3.0, 4.0]);
        let kept = keep_last_rows(vec![rows(3, 0.0), rows(2, 3.0)], Some(1)).unwrap();
        assert_eq!(values(&kept), [4.0]);
        let kept = keep_last_rows(vec![rows(3, 0.0), rows(2, 3.0)], Some(9)).unwrap();
        assert_eq!(values(&kept), [0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn zero_rows_returns_nothing() {
        assert!(keep_last_rows(vec![rows(3, 0.0)], Some(0))
            .unwrap()
            .is_empty());
    }
}
