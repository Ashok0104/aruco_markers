import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ArUco Detector',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: Brightness.light,
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        brightness: Brightness.dark,
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      themeMode: ThemeMode.system,
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  final TextEditingController _serverIpController = TextEditingController();
  String _serverBaseUrl = '';
  bool _isLoading = false;
  Map<String, dynamic>? _detectionResult;
  List<String> _availableDictionaries = [];
  List<String> _selectedDictionaries = [];

  @override
  void initState() {
    super.initState();
    _loadSavedServerIp();
    _checkPermissions();
  }

  Future<void> _loadSavedServerIp() async {
    // Here you could load saved IP from SharedPreferences
    // For now, we'll use a default
    _serverIpController.text = "192.168.4.72:8000";
    _updateServerBaseUrl();
  }

  void _updateServerBaseUrl() {
    if (_serverIpController.text.isNotEmpty) {
      setState(() {
        _serverBaseUrl = 'http://${_serverIpController.text}';
      });
      _fetchAvailableDictionaries();
    }
  }

  Future<void> _checkPermissions() async {
    await [
      Permission.camera,
      Permission.storage,
      Permission.bluetooth,
      Permission.bluetoothConnect,
      Permission.bluetoothScan,
    ].request();
  }

  Future<void> _fetchAvailableDictionaries() async {
    if (_serverBaseUrl.isEmpty) return;

    try {
      final response = await http.get(
        Uri.parse('$_serverBaseUrl/available_dictionaries/'),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _availableDictionaries = List<String>.from(data['dictionaries']);
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to fetch dictionaries: $e')),
      );
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? image = await _picker.pickImage(source: source);
      if (image != null) {
        await _processImage(File(image.path));
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error picking image: $e')),
      );
    }
  }

  Future<void> _processImage(File imageFile) async {
    if (_serverBaseUrl.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter server IP address')),
      );
      return;
    }

    setState(() {
      _isLoading = true;
      _detectionResult = null;
    });

    try {
      // Create multipart request
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$_serverBaseUrl/detect_markers/'),
      );

      // Add file
      request.files.add(await http.MultipartFile.fromPath(
        'file',
        imageFile.path,
      ));

      // Add selected dictionaries if any
      if (_selectedDictionaries.isNotEmpty) {
        _selectedDictionaries.forEach((dict) {
          request.fields['dict_types'] = dict;
        });
      }

      // Send request
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        setState(() {
          _detectionResult = jsonDecode(response.body);
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: ${response.body}')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error processing image: $e')),
      );
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _sendBluetoothCommand() async {
    if (_detectionResult == null || _detectionResult!['result_id'] == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No detection result available')),
      );
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final response = await http.post(
        Uri.parse('$_serverBaseUrl/send_bluetooth/'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'result_id': _detectionResult!['result_id'],
        }),
      );

      if (response.statusCode == 200) {
        final responseData = jsonDecode(response.body);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content: Text('Bluetooth sent: ${responseData['data_sent']}')),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Bluetooth error: ${response.body}')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error sending Bluetooth command: $e')),
      );
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ArUco Marker Detector'),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const SizedBox(height: 16),
                  const SizedBox(height: 16),
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Capture Image',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 16),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                            children: [
                              ElevatedButton.icon(
                                onPressed: () => _pickImage(ImageSource.camera),
                                icon: const Icon(Icons.camera_alt),
                                label: const Text('Camera'),
                              ),
                              ElevatedButton.icon(
                                onPressed: () =>
                                    _pickImage(ImageSource.gallery),
                                icon: const Icon(Icons.photo_library),
                                label: const Text('Gallery'),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                  if (_detectionResult != null) ...[
                    const SizedBox(height: 16),
                    Card(
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                const Text(
                                  'Detection Results',
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                ElevatedButton.icon(
                                  onPressed: _sendBluetoothCommand,
                                  icon: const Icon(Icons.bluetooth),
                                  label: const Text('Upload'),
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.blue,
                                    foregroundColor: Colors.white,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 16),
                            _buildResultDisplay(),
                          ],
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ),
    );
  }

  Widget _buildResultDisplay() {
    if (_detectionResult == null) return const SizedBox.shrink();

    final sequentialOutput =
        _detectionResult!['sequential_output'] ?? 'No output';
    // final dictionaryUsed = _detectionResult!['dictionary_used'] ?? 'Unknown';
    final markers = _detectionResult!['markers'] as List<dynamic>? ?? [];
    final rows = _detectionResult!['rows'] as List<dynamic>? ?? [];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        ResultInfoTile(
          title: 'Sequential Output',
          value: sequentialOutput,
          icon: Icons.text_fields,
          color: Colors.purple,
        ),
        // ResultInfoTile(
        //   title: 'Dictionary Used',
        //   value: dictionaryUsed,
        //   icon: Icons.library_books,
        //   color: Colors.teal,
        // ),
        // ResultInfoTile(
        //   title: 'Markers Detected',
        //   value: markers.length.toString(),
        //   icon: Icons.grid_on,
        //   color: Colors.orange,
        // ),
        // const SizedBox(height: 16),
        // const Text(
        //   'Detected Markers by Row:',
        //   style: TextStyle(fontWeight: FontWeight.bold),
        // ),
        // const SizedBox(height: 8),
        // Container(
        //   decoration: BoxDecoration(
        //     border: Border.all(color: Colors.grey.shade300),
        //     borderRadius: BorderRadius.circular(8),
        //   ),
        //   child: SingleChildScrollView(
        //     scrollDirection: Axis.horizontal,
        //     child: DataTable(
        //       columns: const [
        //         DataColumn(label: Text('Row')),
        //         DataColumn(label: Text('Markers')),
        //       ],
        //       rows: List.generate(rows.length, (index) {
        //         final rowContent = (rows[index] as List).join(', ');
        //         return DataRow(
        //           cells: [
        //             DataCell(Text('${index + 1}')),
        //             DataCell(Text(rowContent)),
        //           ],
        //         );
        //       }),
        //     ),
        //   ),
        // ),
      ],
    );
  }
}

class ResultInfoTile extends StatelessWidget {
  final String title;
  final String value;
  final IconData icon;
  final Color color;

  const ResultInfoTile({
    required this.title,
    required this.value,
    required this.icon,
    required this.color,
    Key? key,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Row(
        children: [
          CircleAvatar(
            radius: 18,
            backgroundColor: color.withOpacity(0.2),
            child: Icon(icon, color: color, size: 18),
          ),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey.shade600,
                ),
              ),
              Text(
                value,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
