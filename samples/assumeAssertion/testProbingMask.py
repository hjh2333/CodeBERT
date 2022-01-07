from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline
# As stated in the paper, CodeBERT is not suitable for mask prediction task,
# while CodeBERT (MLM) is suitable for mask prediction task.

# We give an example on how to use CodeBERT(MLM) for mask prediction task.
model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

CODE = '''public long calculateLargestProductForSeriesLength(int n) {
    if (n < 0)
        throw new IllegalArgumentException(\"Series length must be non-negative.\");
    return IntStream.rangeClosed(0, str.length() - n).mapToObj(index -> str.substring(index, index + n)).mapToLong(s -> s.chars().map(Character::getNumericValue).mapToLong(Long::valueOf).reduce(1, (k, l) -> k * l)).max().orElseThrow(() -> new IllegalArgumentException(\"Series length must be less than or equal to the length of the string to search.\"));
}
public void testCorrectlyCalculatesLargestProductOfLength0ForEmptyStringToSearch(){
    final LargestSeriesProductCalculator calculator = new LargestSeriesProductCalculator(\"\");
    final long expectedProduct = 1;
    final long actualProduct = calculator.calculateLargestProductForSeriesLength(0);
    assertEquals(expectedProduct, actualProduct);
}
public void testCorrectlyCalculatesLargestProductOfLength5ForEmptyStringToSearch(){
    final LargestSeriesProductCalculator calculator = new LargestSeriesProductCalculator(\"\");
    final long expectedProduct = 1;
    final long actualProduct = calculator.calculateLargestProductForSeriesLength(5);
    assertEquals(<mask><mask><mask> <mask><mask><mask>
}'''
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

outputs = fill_mask(CODE)
for entries in outputs:
    for entry in entries:
        print(entry['token_str'], end=', ')
    print('\n')