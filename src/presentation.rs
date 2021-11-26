use regex::Regex;
use std::{collections::HashMap, str::FromStr};

use crate::AppError;

#[derive(Debug, PartialEq, Eq)]
pub struct Word(Vec<(u32, i32)>);

#[derive(Debug, PartialEq, Eq)]
pub struct Presentation {
    num_generators: u32,
    relations: Vec<Word>,
    symbols: Vec<String>,
    symbol_table: HashMap<String, u32>,
}

impl FromStr for Presentation {
    type Err = AppError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut symbols = Vec::new();
        let mut relations = Vec::new();
        let mut symbol_table: HashMap<String, u32> = HashMap::new();
        let mut num_generators: u32 = 0;

        let presentation_regex = Regex::new(r"<(.*)\|(.*)>").unwrap();
        let generator_regex = Regex::new(r"([[:alpha:]][[:alpha:]0-9]*)").unwrap();
        let relation_regex = Regex::new(r"(([[:alpha:]0-9]+(\^(-)?[0-9]+)?(\.)?)+)").unwrap();

        let cap = presentation_regex.captures(s).ok_or(AppError::BadInput)?;
        let generator_string = &cap[1];
        let relation_string = &cap[2];

        for inner_cap in generator_regex.captures_iter(generator_string) {
            let generator = &inner_cap[0];
            if !symbol_table.contains_key(generator) {
                symbols.push(generator.to_string());
                symbol_table.insert(generator.to_string(), num_generators);
                num_generators += 1;
            }
        }

        for inner_cap in relation_regex.captures_iter(relation_string) {
            let relation_string = &inner_cap[0];
            relations.push(parse_word(relation_string, &symbol_table)?);
        }

        Ok(Presentation {
            num_generators,
            relations,
            symbols,
            symbol_table,
        })
    }
}

pub fn parse_word_problem(
    input_string: &str,
    presentation: &Presentation,
) -> Result<(Word, Word), AppError> {
    let mut split = input_string.trim().split(' ');
    let left = split.nth(0).ok_or(AppError::BadInput)?;
    let right = split.nth(0).ok_or(AppError::BadInput)?;
    Ok((
        parse_word(left, &presentation.symbol_table)?,
        parse_word(right, &presentation.symbol_table)?,
    ))
}

fn parse_word(relation: &str, symbol_table: &HashMap<String, u32>) -> Result<Word, AppError> {
    let mut relation_vec: Vec<(u32, i32)> = Vec::new();

    for term in relation.split('.') {
        let mut exp_split = term.split('^');
        let generator = exp_split.nth(0).ok_or(AppError::BadInput)?;
        let index = symbol_table.get(generator).ok_or(AppError::BadInput)?;
        let exponent = match exp_split.nth(0) {
            Some(s) => s.parse::<i32>().map_err(|_| AppError::BadInput)?,
            None => 1,
        };
        relation_vec.push((*index, exponent));
    }

    Ok(Word(relation_vec))
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use super::{Presentation, Word};

    #[test]
    fn test_presentation_parser() {
        let input_string = "<a, b | a.b.a^-1.b^-1>";
        let symbol_table: HashMap<String, u32> = [("a".to_string(), 0), ("b".to_string(), 1)]
            .into_iter()
            .collect();
        let expected_presentation = Presentation {
            num_generators: 2,
            relations: vec![Word(vec![(0, 1), (1, 1), (0, -1), (1, -1)])],
            symbols: vec!["a".to_string(), "b".to_string()],
            symbol_table,
        };
        assert_eq!(
            input_string.parse::<Presentation>().unwrap(),
            expected_presentation
        );

        let input_string = "<a, ab | ab.a^-1.ab^-1>";
        let symbol_table: HashMap<String, u32> = [("a".to_string(), 0), ("ab".to_string(), 1)]
            .into_iter()
            .collect();
        let expected_presentation = Presentation {
            num_generators: 2,
            relations: vec![Word(vec![(1, 1), (0, -1), (1, -1)])],
            symbols: vec!["a".to_string(), "ab".to_string()],
            symbol_table,
        };
        assert_eq!(
            input_string.parse::<Presentation>().unwrap(),
            expected_presentation
        );
    }
}
