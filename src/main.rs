use anyhow::{Context, Result};
use std::io;
use word_problem_cuda::presentation::{parse_word_problem, Presentation};

fn main() -> Result<()> {
    let mut buffer = String::new();
    io::stdin()
        .read_line(&mut buffer)
        .context("Error reading line")?;
    let presentation: Presentation = buffer.parse().context("Error parsing presentation")?;
    println!("Presentation: {:?}", presentation);
    buffer.clear();
    let mut bytes_read = io::stdin()
        .read_line(&mut buffer)
        .context("Error reading line")?;
    while bytes_read > 0 {
        let words = parse_word_problem(&buffer, &presentation);
        println!("Word problem: {:?}", words);

        buffer.clear();
        bytes_read = io::stdin()
            .read_line(&mut buffer)
            .context("Error reading line")?;
    }

    Ok(())
}
