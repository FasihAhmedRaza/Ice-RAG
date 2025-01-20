import { openai } from './open.js';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { OpenAIEmbeddings } from '@langchain/openai';
import { CharacterTextSplitter } from 'langchain/text_splitter';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';

const pdf = './faqs.pdf';

const customPrompt = {
    temperature: 0.5,  // Balanced between accuracy and creativity
    systemPrompt: `You are an expert ice sculptures and ice butchers with decades of experience. Always start with a SHORT, DIRECT answer to the user's question in 1-2 sentences. Then provide optional detailed information using the following guidelines:

                  SHORT ANSWER FORMAT:
                  - First give a clear, direct answer in maximum 2 sentences
                  - Use simple, straightforward language
                  - Focus only on the specific question asked
                  - If possible, include one key number/date/fact
                  
                  DETAILED INFORMATION (only if relevant):
                  - Provide technical information about ice carving techniques, tools, and methods
                  - Include historical context about ice sculpting traditions
                  - Share temperature requirements and ice preparation methods
                  - Explain types of ice (crystal clear, commercial, natural)
                  - Offer practical tips and industry best practices
                  - Discuss safety considerations and proper tool handling
                  - Cover artistic and commercial aspects
                  - Mention specific tools and equipment
                  - Include storage, transportation, and display information
                  - Reference famous sculptors and events
                  
                  Key expertise areas:
                  - Traditional and modern carving techniques
                  - Competition-level sculpting
                  - Commercial ice production
                  - Cold room operations
                  - Tool maintenance
                  - Event planning
                  - Ice block construction
                  - Various styling (Japanese, European, Contemporary)
                  - Display lighting
                  - Seasonal considerations

                  Example format:
                  Q: "What temperature should ice be stored at?"
                  A: Ice sculptures should be stored at -10°F (-23°C) for optimal preservation and working conditions.

               
                  
                  `,
};

// Initialize PDF loader and processing
const initializePDF = async () => {
    try {
        const loader = new PDFLoader(pdf);
        const loadedDoc = await loader.load();
        
        const splitter = new CharacterTextSplitter({
            separator: '. ',
            chunkSize: 5000,
            chunkOverlap: 500
        });
        
        const pdfDocs = await splitter.splitDocuments(loadedDoc);
        const store = await MemoryVectorStore.fromDocuments(
            pdfDocs,
            new OpenAIEmbeddings()
        );
        
        return store;
    } catch (error) {
        console.error('Error loading PDF:', error);
        return null;
    }
};

// Enhanced OpenAI response function with custom prompt
const getOpenAIResponse = async (question) => {
    try {
        const response = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            temperature: customPrompt.temperature,
            messages: [
                {
                    role: 'system',
                    content: customPrompt.systemPrompt
                },
                {
                    role: 'user',
                    content: question
                }
            ]
        });
        
        return {
            content: response.choices[0].message.content,
            source: 'OpenAI General Response'
        };
    } catch (error) {
        console.error('Error getting OpenAI response:', error);
        throw error;
    }
};

// Function to check if PDF results are relevant
const isRelevantPDFContent = async (results, question) => {
    if (results.length === 0) return false;
    
    const response = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        temperature: 0,
        messages: [
            {
                role: 'system',
                content: 'Determine if the provided content contains relevant information to answer the question. Respond with only "true" or "false".'
            },
            {
                role: 'user',
                content: `Question: ${question}
                         Content: ${results.map(r => r.pageContent).join('\n')}
                         Does this content contain specific, relevant information to answer the question?`
            }
        ]
    });
    
    return response.choices[0].message.content.toLowerCase().includes('true');
};

// Main query function
const query = async () => {


    const question = process.argv[2] || ` What temperature should ice be stored at?  `;


 
    try {
        const store = await initializePDF();
        
        if (store) {
            const results = await store.similaritySearch(question, 2);
            // console.log('PDF Search Results:', results); // Debugging information
            const isRelevant = await isRelevantPDFContent(results, question);
            console.log('Is PDF Content Relevant:', isRelevant); // Debugging information
            
            if (isRelevant) {
                const response = await openai.chat.completions.create({
                    model: 'gpt-4',
                    temperature: 0,
                    messages: [
                        {
                            role: 'system',
                            content: 'You are an AI assistant. Provide accurate answers based on the given context. Dont mentioned Based on the context provided'
                        },
                        {
                            role: 'user',
                            content: `Answer the following question using the provided context: 
                                     Question: ${question}
                                     Context: ${results.map((r) => r.pageContent).join('\n')}`
                        }
                    ]
                });
                
                console.log('\n\nAI (From PDF):', response.choices[0].message.content);
                // console.log('\nSource:', results.map((r) => r.metadata.source).join(', '));
                return;
            }
        }
        
        console.log('\nGenerating detailed response from general knowledge...\n');
        const fallbackResponse = await getOpenAIResponse(question);
        console.log('AI (General Response):', fallbackResponse.content);
        console.log('\nSource:', fallbackResponse.source);
        
    } catch (error) {
        console.error('Error processing query:', error);
    }
};

query();